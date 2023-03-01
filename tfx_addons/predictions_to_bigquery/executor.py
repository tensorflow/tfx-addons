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
"""
Executor functionality to write prediction results usually from a BulkInferrer
to BigQuery.
"""

import datetime
import os
from typing import Any, Dict, Generator, List, Tuple

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

from .utils import (create_annotation_fields, feature_to_bq_schema,
                    load_schema, parse_schema)

_SCHEMA_FILE = "schema.pbtxt"
_ADDITIONAL_BQ_PARAMETERS = {}
_DECIMAL_PLACES = 6


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

  def _parse_example(self, serialized: bytes) -> Dict[str, Any]:
    parsed_example = tf.io.parse_example(serialized, self._features)
    output = {}
    for key, tensor in parsed_example.items():
      value_list = tensor.numpy().tolist()
      if isinstance(value_list[0], bytes):
        value_list = [v.decode('utf-8') for v in value_list]
      value = value_list[0] if len(value_list) == 1 else value_list
      output[key] = value
    return output

  def process(self, element) -> Generator[Dict[str, Any], None, None]:
    """Processes element."""
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
  """
    Beam Executor for predictions_to_bq.
    """
  def Do(
      self,
      input_dict: Dict[str, List[types.Artifact]],
      output_dict: Dict[str, List[types.Artifact]],
      exec_properties: Dict[str, Any],
  ) -> None:
    """Do function for predictions_to_bq executor."""

    timestamp = datetime.datetime.now().replace(second=0, microsecond=0)

    # check required executive properties
    if exec_properties["bq_table_name"] is None:
      raise ValueError("bq_table_name must be set in exec_properties")
    if exec_properties["filter_threshold"] is None:
      raise ValueError("filter_threshold must be set in exec_properties")
    if exec_properties["vocab_label_file"] is None:
      raise ValueError("vocab_label_file must be set in exec_properties")

    # get labels from tf transform generated vocab file
    transform_output = artifact_utils.get_single_uri(
        input_dict["transform_graph"])
    tf_transform_output = tft.TFTransformOutput(transform_output)
    tft_vocab = tf_transform_output.vocabulary_by_name(
        vocab_filename=exec_properties["vocab_label_file"])
    labels = [label.decode() for label in tft_vocab]
    logging.info(f"found the following labels from TFT vocab: {labels}")

    # get predictions from predict log
    inference_results_uri = artifact_utils.get_single_uri(
        input_dict["inference_results"])

    # set table prefix and partitioning parameters
    bq_table_name = exec_properties["bq_table_name"]
    if exec_properties["table_suffix"]:
      bq_table_name += "_" + timestamp.strftime(
          exec_properties["table_suffix"])

    if exec_properties["expiration_time_delta"]:
      expiration_time = int(
          timestamp.timestamp()) + exec_properties["expiration_time_delta"]
      _ADDITIONAL_BQ_PARAMETERS.update(
          {"expirationTime": str(expiration_time)})
      logging.info(
          f"expiration time on {bq_table_name} set to {expiration_time}")

    if exec_properties["table_partitioning"]:
      _ADDITIONAL_BQ_PARAMETERS.update({"timePartitioning": {"type": "DAY"}})
      logging.info(f"time partitioning on {bq_table_name} set to DAY")

    # set prediction result file path and decoder
    prediction_log_path = f"{inference_results_uri}/*.gz"
    prediction_log_decoder = beam.coders.ProtoCoder(
        prediction_log_pb2.PredictionLog)

    # get features from tfx schema if present
    if input_dict["schema"]:
      schema_uri = os.path.join(
          artifact_utils.get_single_uri(input_dict["schema"]), _SCHEMA_FILE)
      features = load_schema(schema_uri)

    # generate features from predictions
    else:
      features = parse_schema(prediction_log_path)

    # generate bigquery schema from tfx schema (features)
    bq_schema_fields = feature_to_bq_schema(features, required=True)
    bq_schema_fields.extend(
        create_annotation_fields(label_field_name="category_label",
                                 score_field_name="score",
                                 required=True,
                                 add_datetime_field=True))
    bq_schema = {"fields": bq_schema_fields}
    logging.info(f"generated bq_schema: {bq_schema}")

    with self._make_beam_pipeline() as pipeline:
      _ = (pipeline
           | "Read Prediction Log" >> beam.io.ReadFromTFRecord(
               prediction_log_path, coder=prediction_log_decoder)
           | "Filter and Convert to Dict" >> beam.ParDo(
               FilterPredictionToDictFn(
                   labels=labels,
                   features=features,
                   timestamp=timestamp,
                   filter_threshold=exec_properties["filter_threshold"],
               ))
           | "Write Dict to BQ" >> beam.io.gcp.bigquery.WriteToBigQuery(
               table=bq_table_name,
               schema=bq_schema,
               additional_bq_parameters=_ADDITIONAL_BQ_PARAMETERS,
               create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
               write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
           ))

    bigquery_export = artifact_utils.get_single_instance(
        output_dict["bigquery_export"])

    bigquery_export.set_string_custom_property("generated_bq_table_name",
                                               bq_table_name)

    logging.info(f"Annotated data exported to {bq_table_name}")
