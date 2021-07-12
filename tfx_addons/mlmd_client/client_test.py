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
