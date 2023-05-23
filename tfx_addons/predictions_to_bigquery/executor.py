# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This code was originally written by Hannes Hapke (Digits Financial Inc.)
# on Feb. 6, 2023.
"""Implements executor to write BulkInferrer prediction results to BigQuery."""

import datetime
import os
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import apache_beam as beam
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from tensorflow_serving.apis import prediction_log_pb2
from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.types import artifact_utils

# TODO(cezequiel): Move relevant functions in utils module here.
from tfx_addons.predictions_to_bigquery import utils

_SCHEMA_FILE_NAME = "schema.pbtxt"
_DECIMAL_PLACES = 6
_DEFAULT_TIMESTRING_FORMAT = '%Y%m%d_%H%M%S'
_REQUIRED_EXEC_PROPERTIES = (
    'bq_table_name',
    'bq_dataset',
    'filter_threshold',
    'gcp_project',
    'gcs_temp_dir',
    'vocab_label_file',
)
_REGEX_CHARS_TO_REPLACE = re.compile(r'[^a-zA-Z0-9_]')


def _check_exec_properties(exec_properties: Mapping[str, Any]) -> None:
  for key in _REQUIRED_EXEC_PROPERTIES:
    if exec_properties[key] is None:
      raise ValueError(f'{key} must be set in exec_properties')


def _get_labels(transform_output_uri: str, vocab_file: str) -> Sequence[str]:
  tf_transform_output = tft.TFTransformOutput(transform_output_uri)
  tft_vocab = tf_transform_output.vocabulary_by_name(vocab_filename=vocab_file)
  return [label.decode() for label in tft_vocab]


def _get_bq_table_name(
    basename: str,
    timestamp: Optional[datetime.datetime] = None,
    timestring_format: Optional[str] = None,
) -> str:
  if timestamp is not None:
    timestring_format = timestring_format or _DEFAULT_TIMESTRING_FORMAT
    return basename + '_' + timestamp.strftime(timestring_format)
  return basename


def _get_additional_bq_parameters(
    expiration_days: Optional[int] = None,
    table_partitioning: bool = False,
) -> Mapping[str, Any]:
  output = {}
  if table_partitioning:
    time_partitioning = {'type': 'DAY'}
    logging.info('BigQuery table time partitioning set to DAY')
    if expiration_days:
      expiration_time_delta = datetime.timedelta(days=expiration_days)
      expiration_milliseconds = expiration_time_delta.total_seconds() * 1000
      logging.info(
          f'BigQuery table partition expiration time set to {expiration_days}'
          ' days')
      time_partitioning['expirationMs'] = expiration_milliseconds
    output['timePartitioning'] = time_partitioning
  return output


def _get_features(
    *,
    schema_uri: Optional[str] = None,
    prediction_log_path: Optional[str] = None,
) -> Mapping[str, Any]:
  if schema_uri:
    schema_file = os.path.join(schema_uri, _SCHEMA_FILE_NAME)
    return utils.load_schema(schema_file)

  if not prediction_log_path:
    raise ValueError('Specify one of `schema_uri` or `prediction_log_path`.')

  return utils.parse_schema(prediction_log_path)


def _get_bq_field_name_from_key(key: str) -> str:
  field_name = _REGEX_CHARS_TO_REPLACE.sub('_', key)
  return re.sub('_+', '_', field_name).strip('_')


def _features_to_bq_schema(features: Mapping[str, Any],
                           required: bool = False):
  bq_schema_fields_ = utils.feature_to_bq_schema(features, required=required)
  bq_schema_fields = []
  for field in bq_schema_fields_:
    field['name'] = _get_bq_field_name_from_key(field['name'])
    bq_schema_fields.append(field)
  bq_schema_fields.extend(
      utils.create_annotation_fields(label_field_name="category_label",
                                     score_field_name="score",
                                     required=required,
                                     add_datetime_field=True))
  return {"fields": bq_schema_fields}


def _tensor_to_native_python_value(
    tensor: Union[tf.Tensor, tf.sparse.SparseTensor]
) -> Optional[Union[int, float, str]]:
  """Converts a TF Tensor to a native Python value."""
  if isinstance(tensor, tf.sparse.SparseTensor):
    values = tensor.values.numpy()
  else:
    values = tensor.numpy()
  if not values:
    return None
  values = np.squeeze(values)  # Removes extra dimension, e.g. shape (n, 1).
  values = values.item()  # Converts to native Python type
  if isinstance(values, Sequence) and isinstance(values[0], bytes):
    return [v.decode('utf-8') for v in values]
  if isinstance(values, bytes):
    return values.decode('utf-8')
  return values


@beam.typehints.with_input_types(str)
@beam.typehints.with_output_types(beam.typehints.Iterable[Tuple[str, str,
                                                                Any]])
class FilterPredictionToDictFn(beam.DoFn):
  """Converts a PredictionLog proto to a dict."""
  def __init__(
      self,
      labels: List,
      features: Any,
      timestamp: datetime.datetime,
      filter_threshold: float,
      score_multiplier: float = 1.,
  ):
    super().__init__()
    self._labels = labels
    self._features = features
    self._filter_threshold = filter_threshold
    self._score_multiplier = score_multiplier
    self._timestamp = timestamp

  def _parse_prediction(self, predictions: npt.ArrayLike):
    prediction_id = np.argmax(predictions)
    logging.debug("Prediction id: %s", prediction_id)
    logging.debug("Predictions: %s", predictions)
    label = self._labels[prediction_id]
    score = predictions[0][prediction_id]
    return label, score

  def _parse_example(self, serialized: bytes) -> Mapping[str, Any]:
    parsed_example = tf.io.parse_example(serialized, self._features)
    output = {}
    for key, tensor in parsed_example.items():
      field = _get_bq_field_name_from_key(key)
      value = _tensor_to_native_python_value(tensor)
      # To add a null value to BigQuery from JSON, omit the key,value pair
      # with null value.
      if value is None:
        continue
      output[field] = value
    return output

  def process(self, element, *args, **kwargs):  # pylint: disable=missing-function-docstring
    del args, kwargs  # unused

    parsed_prediction_scores = tf.make_ndarray(
        element.predict_log.response.outputs["outputs"])
    label, score = self._parse_prediction(parsed_prediction_scores)
    if score >= self._filter_threshold:
      output = {
          "category_label": label,
          # Workaround to issue with the score value having additional non-zero values
          # in higher decimal places.
          # e.g. 0.8 -> 0.800000011920929
          "score": round(score * self._score_multiplier, _DECIMAL_PLACES),
          "datetime": self._timestamp,
      }
      output.update(
          self._parse_example(
              element.predict_log.request.inputs['examples'].string_val))
      yield output


class Executor(base_beam_executor.BaseBeamExecutor):
  """Implements predictions-to-bigquery component logic."""
  def Do(
      self,
      input_dict: Mapping[str, List[types.Artifact]],
      output_dict: Mapping[str, List[types.Artifact]],
      exec_properties: Mapping[str, Any],
  ) -> None:
    """Do function for predictions_to_bq executor."""

    timestamp = datetime.datetime.now().replace(second=0, microsecond=0)

    # Check required keys set in exec_properties
    _check_exec_properties(exec_properties)

    # get labels from tf transform generated vocab file
    labels = _get_labels(
        artifact_utils.get_single_uri(input_dict['transform_graph']),
        exec_properties['vocab_label_file'],
    )
    logging.info(f"found the following labels from TFT vocab: {labels}")

    # set BigQuery table name and timestamp suffix if specified.
    bq_table_name = _get_bq_table_name(exec_properties['bq_table_name'],
                                       timestamp,
                                       exec_properties['table_suffix'])

    # set prediction result file path and decoder
    inference_results_uri = artifact_utils.get_single_uri(
        input_dict["inference_results"])
    prediction_log_path = f"{inference_results_uri}/*.gz"
    prediction_log_decoder = beam.coders.ProtoCoder(
        prediction_log_pb2.PredictionLog)

    # get schema features
    features = _get_features(schema_uri=artifact_utils.get_single_uri(
        input_dict["schema"]),
                             prediction_log_path=prediction_log_path)

    # generate bigquery schema from tf transform features
    bq_schema = _features_to_bq_schema(features)
    logging.info(f'generated bq_schema: {bq_schema}.')

    additional_bq_parameters = _get_additional_bq_parameters(
        exec_properties.get('expiration_time_delta'),
        exec_properties.get('table_partitioning'))

    # run the Beam pipeline to write the inference data to bigquery
    with self._make_beam_pipeline() as pipeline:
      _ = (pipeline
           | 'Read Prediction Log' >> beam.io.ReadFromTFRecord(
               prediction_log_path, coder=prediction_log_decoder)
           | 'Filter and Convert to Dict' >> beam.ParDo(
               FilterPredictionToDictFn(
                   labels=labels,
                   features=features,
                   timestamp=timestamp,
                   filter_threshold=exec_properties['filter_threshold']))
           | 'Write Dict to BQ' >> beam.io.gcp.bigquery.WriteToBigQuery(
               table=bq_table_name,
               dataset=exec_properties['bq_dataset'],
               project=exec_properties['gcp_project'],
               schema=bq_schema,
               additional_bq_parameters=additional_bq_parameters,
               create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
               write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
               custom_gcs_temp_location=exec_properties['gcs_temp_dir']))

    bigquery_export = artifact_utils.get_single_instance(
        output_dict['bigquery_export'])
    bigquery_export.set_string_custom_property('generated_bq_table_name',
                                               bq_table_name)
    logging.info(f'Annotated data exported to {bq_table_name}')
