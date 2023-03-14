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
"""
Digits Prediction-to-BigQuery: Functionality to write prediction results usually
 from a BulkInferrer to BigQuery.
"""

from typing import Optional

from tfx import types
from tfx.dsl.components.base import base_component, executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter

from .executor import Executor as AnnotateUnlabeledCategoryDataExecutor

_MIN_THRESHOLD = 0.8
_VOCAB_FILE = "vocab_label_txt"

# pylint: disable=missing-class-docstring


class AnnotateUnlabeledCategoryDataComponentSpec(types.ComponentSpec):

  PARAMETERS = {
      # These are parameters that will be passed in the call to
      # create an instance of this component.
      "vocab_label_file": ExecutionParameter(type=str),
      "bq_table_name": ExecutionParameter(type=str),
      "filter_threshold": ExecutionParameter(type=float),
      "table_suffix": ExecutionParameter(type=str),
      "table_partitioning": ExecutionParameter(type=bool),
      "expiration_time_delta": ExecutionParameter(type=int),
  }
  INPUTS = {
      # This will be a dictionary with input artifacts, including URIs
      "transform_graph":
      ChannelParameter(type=standard_artifacts.TransformGraph),
      "inference_results":
      ChannelParameter(type=standard_artifacts.InferenceResult),
      "schema":
      ChannelParameter(type=standard_artifacts.Schema),
  }
  OUTPUTS = {
      "bigquery_export": ChannelParameter(type=standard_artifacts.String),
  }


class AnnotateUnlabeledCategoryDataComponent(base_component.BaseComponent):
  """
    AnnotateUnlabeledCategoryData Component.

    The component takes the following input artifacts:
    * Inference results: InferenceResult
    * Transform graph: TransformGraph
    * Schema: Schema (optional) if not present, the component will determine
    the schema (only predtion supported at the moment)

    The component takes the following parameters:
    * vocab_label_file: str - The file name of the file containing the
      vocabulary labels (produced by TFT).
    * bq_table_name: str - The name of the BigQuery table to write the results
      to.
    * filter_threshold: float   - The minimum probability threshold for a
      prediction to be considered a positive, thrustworthy prediction.
      Default is 0.8.
    * table_suffix: str (optional) - If provided, the generated datetime string
      will be added the BigQuery table name as suffix. The default is %Y%m%d.
    * table_partitioning: bool - Whether to partition the table by DAY. If True,
      the generated BigQuery table will be partition by date. If False, no
      partitioning will be applied. Default is True.
    * expiration_time_delta: int (optional) - The number of seconds after which
      the table will expire.

    The component produces the following output artifacts:
    * bigquery_export: String - The URI of the BigQuery table containing the
      results.
    """

  SPEC_CLASS = AnnotateUnlabeledCategoryDataComponentSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(
      AnnotateUnlabeledCategoryDataExecutor)

  def __init__(
      self,
      inference_results: types.Channel = None,
      transform_graph: types.Channel = None,
      bq_table_name: str = None,
      vocab_label_file: str = _VOCAB_FILE,
      filter_threshold: float = _MIN_THRESHOLD,
      table_suffix: str = "%Y%m%d",
      table_partitioning: bool = True,
      schema: Optional[types.Channel] = None,
      expiration_time_delta: Optional[int] = 0,
      bigquery_export: Optional[types.Channel] = None,
  ):

    bigquery_export = bigquery_export or types.Channel(
        type=standard_artifacts.String)
    schema = schema or types.Channel(type=standard_artifacts.Schema())

    spec = AnnotateUnlabeledCategoryDataComponentSpec(
        inference_results=inference_results,
        transform_graph=transform_graph,
        schema=schema,
        bq_table_name=bq_table_name,
        vocab_label_file=vocab_label_file,
        filter_threshold=filter_threshold,
        table_suffix=table_suffix,
        table_partitioning=table_partitioning,
        expiration_time_delta=expiration_time_delta,
        bigquery_export=bigquery_export,
    )
    super().__init__(spec=spec)
