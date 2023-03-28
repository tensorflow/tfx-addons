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
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import apache_beam as beam
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from tensorflow_serving.apis import prediction_log_pb2
from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.types import Artifact, artifact_utils

from tfx_addons.predictions_to_bigquery import utils

_DECIMAL_PLACES = 6
_DEFAULT_TIMESTRING_FORMAT = '%Y%m%d_%H%M%S'
_REQUIRED_EXEC_PROPERTIES = (
    'bq_table_name',
    'filter_threshold',
    'gcs_temp_dir',
)
_REGEX_BQ_TABLE_NAME = re.compile(r'^[\w-]*:?[\w_]+\.[\w_]+$')


def _check_exec_properties(exec_properties: Dict[str, Any]) -> None:
  for key in _REQUIRED_EXEC_PROPERTIES:
    if exec_properties[key] is None:
      raise ValueError(f'{key} must be set in exec_properties')


def _get_prediction_log_path(inference_results: List[Artifact]) -> str:
  inference_results_uri = artifact_utils.get_single_uri(inference_results)
  return f'{inference_results_uri}/*.gz'


def _get_tft_output(
    transform_graph: Optional[List[Artifact]] = None
) -> Optional[tft.TFTransformOutput]:
  if not transform_graph:
    return None

  transform_graph_uri = artifact_utils.get_single_uri(transform_graph)
  return tft.TFTransformOutput(transform_graph_uri)


def _get_labels(tft_output: tft.TFTransformOutput,
                vocab_file: str) -> List[str]:
  tft_vocab = tft_output.vocabulary_by_name(vocab_filename=vocab_file)
  return [label.decode() for label in tft_vocab]


def _check_bq_table_name(bq_table_name: str) -> None:
  if _REGEX_BQ_TABLE_NAME.match(bq_table_name) is None:
    raise ValueError('Invalid BigQuery table name.'
                     ' Specify in either `PROJECT:DATASET.TABLE` or'
                     ' `DATASET.TABLE` format.')


def _add_bq_table_name_suffix(basename: str,
                              timestamp: Optional[datetime.datetime] = None,
                              timestring_format: Optional[str] = None) -> str:
  if timestamp is not None:
    timestring_format = timestring_format or _DEFAULT_TIMESTRING_FORMAT
    return basename + '_' + timestamp.strftime(timestring_format)
  return basename


def _get_additional_bq_parameters(
    table_expiration_days: Optional[int] = None,
    table_partitioning: Optional[bool] = False,
) -> Dict[str, Any]:
  output = {}
  if table_partitioning:
    time_partitioning = {'type': 'DAY'}
    logging.info('BigQuery table time partitioning set to DAY')
    if table_expiration_days:
      expiration_time_delta = datetime.timedelta(days=table_expiration_days)
      expiration_milliseconds = expiration_time_delta.total_seconds() * 1000
      logging.info(
          f'BigQuery table expiration set to {table_expiration_days} days.')
      time_partitioning['expirationMs'] = expiration_milliseconds
    output['timePartitioning'] = time_partitioning
  return output


def _tensor_to_native_python_value(
    tensor: Union[tf.Tensor, tf.sparse.SparseTensor]) -> Optional[Any]:
  """Converts a TF Tensor to a native Python value."""
  if isinstance(tensor, tf.sparse.SparseTensor):
    values = tensor.values.numpy()
  else:
    values = tensor.numpy()
  if not values:
    return None
  values = np.squeeze(values)  # Removes extra dimension, e.g. shape (n, 1).
  values = values.item()  # Converts to native Python type
  if isinstance(values, list) and isinstance(values[0], bytes):
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
      features: Dict[str, tf.io.FixedLenFeature],
      timestamp: datetime.datetime,
      filter_threshold: float,
      labels: Optional[List[str]] = None,
      score_multiplier: float = 1.,
  ):
    super().__init__()
    self._features = features
    self._timestamp = timestamp
    self._filter_threshold = filter_threshold
    self._labels = labels
    self._score_multiplier = score_multiplier

  def _parse_prediction(
      self, predictions: npt.ArrayLike) -> Tuple[Optional[str], float]:
    prediction_id = np.argmax(predictions)
    logging.debug("Prediction id: %s", prediction_id)
    logging.debug("Predictions: %s", predictions)
    label = self._labels[prediction_id] if self._labels is not None else None
    score = predictions[0][prediction_id]
    return label, score

  def _parse_example(self, serialized: bytes) -> Dict[str, Any]:
    parsed_example = tf.io.parse_example(serialized, self._features)
    output = {}
    for key, tensor in parsed_example.items():
      value = _tensor_to_native_python_value(tensor)
      # To add a null value to BigQuery from JSON, omit the key,value pair
      # with null value.
      if value is None:
        continue
      field = utils.get_bq_field_name_from_key(key)
      output[field] = value
    return output

  def process(self, element, *args, **kwargs):  # pylint: disable=missing-function-docstring
    del args, kwargs  # unused

    parsed_prediction_scores = tf.make_ndarray(
        element.predict_log.response.outputs['outputs'])
    label, score = self._parse_prediction(parsed_prediction_scores)
    if score >= self._filter_threshold:
      output = {
          # Workaround to issue with the score value having additional non-zero values
          # in higher decimal places.
          # e.g. 0.8 -> 0.800000011920929
          'score': round(score * self._score_multiplier, _DECIMAL_PLACES),
          'datetime': self._timestamp,
      }
      if label is not None:
        output['category_label'] = label
      output.update(
          self._parse_example(
              element.predict_log.request.inputs['examples'].string_val))
      yield output


class Executor(base_beam_executor.BaseBeamExecutor):
  """Implements predictions-to-bigquery component logic."""
  def Do(
      self,
      input_dict: Dict[str, List[types.Artifact]],
      output_dict: Dict[str, List[types.Artifact]],
      exec_properties: Dict[str, Any],
  ) -> None:
    """Do function for predictions_to_bq executor."""

    # Check required keys set in exec_properties
    _check_exec_properties(exec_properties)

    # Get prediction log file path and decoder
    prediction_log_path = _get_prediction_log_path(
        input_dict['inference_results'])
    prediction_log_decoder = beam.coders.ProtoCoder(
        prediction_log_pb2.PredictionLog)

    tft_output = _get_tft_output(input_dict.get('transform_graph'))

    # get schema features
    features = utils.get_feature_spec(
        schema=input_dict.get('schema'),
        tft_output=tft_output,
        prediction_log_path=prediction_log_path,
    )

    # get label names from TFTransformOutput object, if applicable
    if tft_output is not None and 'vocab_label_file' in exec_properties:
      label_key = exec_properties['vocab_label_file']
      labels = _get_labels(tft_output, label_key)
      logging.info(f'Found the following labels from TFT vocab: {labels}.')
      _ = features.pop(label_key, None)
    else:
      labels = None
      logging.info('No TFTransform output given; no labels parsed.')

    # set BigQuery table name and timestamp suffix if specified.
    _check_bq_table_name(exec_properties['bq_table_name'])
    timestamp = datetime.datetime.now().replace(second=0, microsecond=0)
    bq_table_name = _add_bq_table_name_suffix(
        exec_properties['bq_table_name'], timestamp,
        exec_properties.get('table_time_suffix'))

    # generate bigquery schema from tf transform features
    add_label_field = labels is not None
    bq_schema = utils.feature_spec_to_bq_schema(
        features, add_label_field=add_label_field)
    logging.info(f'generated bq_schema: {bq_schema}.')

    additional_bq_parameters = _get_additional_bq_parameters(
        exec_properties.get('table_expiration_days'),
        exec_properties.get('table_partitioning'))

    # run the Beam pipeline to write the inference data to bigquery
    with self._make_beam_pipeline() as pipeline:
      _ = (pipeline
           | 'Read Prediction Log' >> beam.io.ReadFromTFRecord(
               prediction_log_path, coder=prediction_log_decoder)
           | 'Filter and Convert to Dict' >> beam.ParDo(
               FilterPredictionToDictFn(
                   features=features,
                   timestamp=timestamp,
                   filter_threshold=exec_properties['filter_threshold'],
                   labels=labels))
           | 'Write Dict to BQ' >> beam.io.WriteToBigQuery(
               table=bq_table_name,
               schema=bq_schema,
               additional_bq_parameters=additional_bq_parameters,
               create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
               write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
               custom_gcs_temp_location=exec_properties['gcs_temp_dir']))

    bigquery_export = artifact_utils.get_single_instance(
        output_dict['bigquery_export'])
    bigquery_export.set_string_custom_property('generated_bq_table_name',
                                               bq_table_name)
    with tf.io.gfile.GFile(bigquery_export.uri, 'w') as output_file:
      output_file.write(bq_table_name)
    logging.info(f'Annotated data exported to {bq_table_name}')
