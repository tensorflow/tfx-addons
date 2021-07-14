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
"""Tests for tfx_addons.mlmd_client.client."""
import os

from ml_metadata.proto import metadata_store_pb2
from tfx.dsl.component.experimental.annotations import (OutputArtifact,
                                                        Parameter)
from tfx.dsl.component.experimental.decorators import component
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration.pipeline import Pipeline
from tfx.types.standard_artifacts import String

from tfx_addons.mlmd_client import client


@component
def print_component(word: Parameter[str], word_out: OutputArtifact[String]):
  print(word)
  word_out.value = word


def _create_pipeline(root_dir: str):
  comp = print_component(word="test")
  connection_config = metadata_store_pb2.ConnectionConfig()
  connection_config.sqlite.filename_uri = os.path.join(root_dir, "db.sqlite")
  connection_config.sqlite.connection_mode = 3  # READWRITE_OPENCREATE
  return Pipeline(
      pipeline_root=root_dir,
      pipeline_name="client_test",
      metadata_connection_config=connection_config,
      components=[comp],
  )


def test_pipeline_exists(tmpdir):
  pipeline = _create_pipeline(tmpdir.mkdir("test").strpath)
  LocalDagRunner().run(pipeline)
  p = client.MetadataClient.from_pipeline(pipeline)
  assert isinstance(p, client.PipelineContext)


def test_get_artifacts(tmpdir):
  pipeline = _create_pipeline(tmpdir.mkdir("test").strpath)
  LocalDagRunner().run(pipeline)
  p = client.MetadataClient.from_pipeline(pipeline)
  assert isinstance(p, client.PipelineContext)
  assert len(p.get_artifact_by_type_name('String')) == 1
