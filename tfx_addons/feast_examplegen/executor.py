# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
Generic TFX FeastExampleGen executor.
"""

import os
from typing import Any, Dict, Optional

import apache_beam as beam
import feast
from absl import logging
from apache_beam.options import value_provider
from feast.infra.offline_stores.bigquery import BigQueryRetrievalJob
from feast.infra.offline_stores.offline_store import RetrievalJob
from google.protobuf import json_format
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from tfx.components.example_gen import base_example_gen_executor
from tfx.extensions.google_cloud_big_query import utils
from tfx.proto import example_gen_pb2

from tfx_addons.feast_examplegen import converters

_REPO_CONFIG_KEY = "repo_conf"
_FEATURE_KEY = "feature_refs"
_FEATURE_SERVICE_KEY = "feature_service_ref"


def _load_custom_config(custom_config):
  config = example_gen_pb2.CustomConfig()
  json_format.Parse(custom_config, config)
  s = Struct()
  config.custom_config.Unpack(s)
  config_dict = MessageToDict(s)
  return config_dict


def _load_feast_feature_store(
    custom_config: Dict[str, Any]) -> feast.FeatureStore:
  repo_config = feast.RepoConfig.parse_raw(custom_config[_REPO_CONFIG_KEY])
  return feast.FeatureStore(config=repo_config)


def _get_retrieval_job(entity_query: str,
                       custom_config: Dict[str, Any]) -> RetrievalJob:
  """Get feast retrieval job

    Args:
        entity_query (str): entity query.
        custom_config (Dict[str, Any]): Custom configuration from component

    Raises:
        RuntimeError: [description]

    Returns:
        RetrievalJob: [description]
    """
  feature_list = custom_config.get(_FEATURE_KEY, None)
  feature_service = custom_config.get(_FEATURE_SERVICE_KEY, None)
  fs = _load_feast_feature_store(custom_config)

  if feature_list:
    features = feature_list
  elif feature_service:
    features = fs.get_feature_service(feature_service)
  else:
    raise RuntimeError(
        "Either feature service or feature list should be provided")

  return fs.get_historical_features(entity_df=entity_query, features=features)


def _get_gcp_project(exec_properties: Dict[str, Any]) -> Optional[str]:
  # Get GCP project from exec_properties.
  beam_pipeline_args = exec_properties["_beam_pipeline_args"]
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      beam_pipeline_args)

  # Try to parse the GCP project ID from the beam pipeline options.
  project = pipeline_options.view_as(
      beam.options.pipeline_options.GoogleCloudOptions).project
  if isinstance(project, value_provider.ValueProvider):
    return project.get()
  return project or None


def _get_datasource_converter(exec_properties: Dict[str, Any],
                              split_pattern: str):
  # Load custom config dictionary
  custom_config = _load_custom_config(exec_properties["custom_config"])

  # Get Feast retrieval job
  retrieval_job = _get_retrieval_job(entity_query=split_pattern,
                                     custom_config=custom_config)

  # Setup datasource and converter.
  if isinstance(retrieval_job, BigQueryRetrievalJob):
    table = retrieval_job.to_bigquery()
    query = f'SELECT * FROM {table}'
    # Internally Beam creates a temporary table and exports from the query.
    datasource = utils.ReadFromBigQuery(query=query)
    converter = converters._BigQueryConverter(  # pylint: disable=protected-access
        query, _get_gcp_project(exec_properties))
  else:
    raise NotImplementedError(
        f"Support for {type(retrieval_job)} is not available yet. "
        "For now we only support BigQuery source.")

  return datasource, converter


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(bytes)
def _FeastToExampleTransform(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, exec_properties: Dict[str, Any],
    split_pattern: str) -> beam.pvalue.PCollection:
  """Read from BigQuery and transform to TF examples.

    Args:
      pipeline: beam pipeline.
      exec_properties: A dict of execution properties.
      split_pattern: Split.pattern in Input config, a BigQuery sql string.

    Returns:
      PCollection of TF examples.
    """
  # Feast doesn't allow us to configure GCP project used to explore BQ (afaik),
  # we instead set the beam project as default in environment as sometimes
  # it may not work on the default project
  gcp_project = _get_gcp_project(exec_properties)
  logging.info("Detected GCP project %s", gcp_project)
  restore_project = None
  if gcp_project:
    logging.info("Overwriting GOOGLE_CLOUD_PROJECT env var to %s", gcp_project)
    restore_project = os.environ.get("GOOGLE_CLOUD_PROJECT", None)
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project
  try:
    datasource, converter = _get_datasource_converter(
        exec_properties=exec_properties, split_pattern=split_pattern)
  finally:
    if restore_project:
      os.environ["GOOGLE_CLOUD_PROJECT"] = restore_project

  # Setup converter from dictionary of str -> value to bytes
  map_function = None
  out_format = exec_properties.get("output_data_format",
                                   example_gen_pb2.FORMAT_TF_EXAMPLE)
  if out_format == example_gen_pb2.FORMAT_TF_EXAMPLE:
    map_function = converter.RowToExampleBytes
  elif out_format == example_gen_pb2.FORMAT_TF_SEQUENCE_EXAMPLE:
    map_function = converter.RowToSequenceExampleBytes
  else:
    raise NotImplementedError(
        f"Format {out_format} is not currently supported."
        " Currently we only support tfexample")

  # Setup pipeline
  return (pipeline
          | "DataRetrieval" >> datasource
          | "ToBytes" >> beam.Map(map_function))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX FeastExampleGen executor."""
  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for Feast to bytes."""
    return _FeastToExampleTransform
