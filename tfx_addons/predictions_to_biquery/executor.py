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
# ToDo(gcasassaez): Fix up linter issues
# pylint: skip-file
"""
Executor functionality to write prediction results usually from a BulkInferrer to BigQuery.
"""

import datetime
import os
from typing import Any, Dict, List, Tuple

import apache_beam as beam
import numpy as np
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from tensorflow.python.eager.context import eager_mode
from tensorflow_serving.apis import prediction_log_pb2
from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.types import artifact_utils

from .utils import (convert_single_value_to_native_py_value,
                    create_annotation_fields, feature_to_bq_schema,
                    load_schema, parse_schema)

_SCORE_MULTIPLIER = 1e6
_SCHEMA_FILE = "schema.pbtxt"
_ADDITIONAL_BQ_PARAMETERS = {}


@beam.typehints.with_input_types(str)
@beam.typehints.with_output_types(beam.typehints.Iterable[Tuple[str, str,
                                                                Any]])
class FilterPredictionToDictFn(beam.DoFn):
  """
    Convert a prediction to a dictionary.
    """
  def __init__(
      self,
      labels: List,
      features: Any,
      ts: datetime.datetime,
      filter_threshold: float,
      score_multiplier: int = _SCORE_MULTIPLIER,
  ):
    self.labels = labels
    self.features = features
    self.filter_threshold = filter_threshold
    self.score_multiplier = score_multiplier
    self.ts = ts

  def _fix_types(self, example):
    with eager_mode():
      return [
          convert_single_value_to_native_py_value(v) for v in example.values()
      ]

  def _parse_prediction(self, predictions):
    prediction_id = np.argmax(predictions)
    logging.debug("Prediction id: %s", prediction_id)
    logging.debug("Predictions: %s", predictions)
    label = self.labels[prediction_id]
    score = predictions[0][prediction_id]
    return label, score

  def process(self, element):
    parsed_examples = tf.make_ndarray(
        element.predict_log.request.inputs["examples"])
    parsed_predictions = tf.make_ndarray(
        element.predict_log.response.outputs["output_0"])

    example_values = self._fix_types(
        tf.io.parse_single_example(parsed_examples[0], self.features))
    label, score = self._parse_prediction(parsed_predictions)

    if score > self.filter_threshold:
      yield {
          # TODO: features should be read dynamically
          "feature0": example_values[0],
          "feature1": example_values[1],
          "feature2": example_values[2],
          "category_label": label,
          "score": int(score * self.score_multiplier),
          "datetime": self.ts,
      }


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

    ts = datetime.datetime.now().replace(second=0, microsecond=0)

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
      bq_table_name += "_" + ts.strftime(exec_properties["table_suffix"])

    if exec_properties["expiration_time_delta"]:
      expiration_time = int(
          ts.timestamp()) + exec_properties["expiration_time_delta"]
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
                   ts=ts,
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
