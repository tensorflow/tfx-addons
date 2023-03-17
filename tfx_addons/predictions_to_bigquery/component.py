# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This code was originally written by Hannes Hapke (Digits Financial Inc.)
# on Feb. 6, 2023.
"""Predictions-to-bigquery component spec."""

from typing import Optional

from tfx import types
from tfx.dsl.components.base import base_component, executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter

from tfx_addons.predictions_to_bigquery import executor

_MIN_THRESHOLD = 0.5

# pylint: disable=missing-class-docstring


class PredictionsToBigQueryComponentSpec(types.ComponentSpec):

  PARAMETERS = {
      'bq_table_name': ExecutionParameter(type=str),
      'gcs_temp_dir': ExecutionParameter(type=str),
      'table_expiration_days': ExecutionParameter(type=int),
      'filter_threshold': ExecutionParameter(type=float),
      'table_partitioning': ExecutionParameter(type=bool),
      'table_time_suffix': ExecutionParameter(type=str, optional=True),
      'vocab_label_file': ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      'inference_results':
      (ChannelParameter(type=standard_artifacts.InferenceResult)),
      'schema': (ChannelParameter(type=standard_artifacts.Schema,
                                  optional=True)),
      'transform_graph':
      (ChannelParameter(type=standard_artifacts.TransformGraph,
                        optional=True)),
  }
  OUTPUTS = {
      'bigquery_export': ChannelParameter(type=standard_artifacts.String),
  }


class PredictionsToBigQueryComponent(base_component.BaseComponent):

  SPEC_CLASS = PredictionsToBigQueryComponentSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

  def __init__(
      self,
      inference_results: types.Channel,
      bq_table_name: str,
      gcs_temp_dir: str,
      bigquery_export: Optional[types.Channel] = None,
      transform_graph: Optional[types.Channel] = None,
      schema: Optional[types.Channel] = None,
      table_expiration_days: Optional[int] = 0,
      filter_threshold: float = _MIN_THRESHOLD,
      table_partitioning: bool = True,
      table_time_suffix: Optional[str] = None,
      vocab_label_file: Optional[str] = None,
  ) -> None:
    """Initialize the component.

    Args:
      inference_results: Inference results channel.
      bq_table_name: BigQuery table name in either PROJECT:DATASET.TABLE.
        or DATASET.TABLE formats.
      bigquery_export: Outputs channel containing generated BigQuery table name.
        The outputted name may contain a timestamp suffix defined by
        `table_suffix`.
      transform_graph: TFTransform graph channel.
        If specified, and `schema` is not specified, the prediction
        input schema shall be derived from this channel.
      schema: Schema channel.
        If specified, the prediction input schema shall be derived from this
        channel.
      expiration_days: BigQuery table expiration in number of days from
        current time. If not specified, the table does not expire by default.
      filter_threshold: Prediction threshold to use to filter prediction scores.
        Keep scores that exceed this threshold.
      table_partitioning: If True, partition table.
        See: https://cloud.google.com/bigquery/docs/partitioned-tables
      table_time_suffix: Time format for table suffix in Linux strftime format.
        Example: '%Y%m%d
      vocab_label_file: Name of the TF transform vocabulary file for the label.
    """
    bigquery_export = bigquery_export or types.Channel(
        type=standard_artifacts.String)
    schema = schema or types.Channel(type=standard_artifacts.Schema)

    spec = PredictionsToBigQueryComponentSpec(
        inference_results=inference_results,
        bq_table_name=bq_table_name,
        gcs_temp_dir=gcs_temp_dir,
        bigquery_export=bigquery_export,
        transform_graph=transform_graph,
        schema=schema,
        table_expiration_days=table_expiration_days,
        filter_threshold=filter_threshold,
        table_partitioning=table_partitioning,
        table_time_suffix=table_time_suffix,
        vocab_label_file=vocab_label_file,
    )
    super().__init__(spec=spec)
