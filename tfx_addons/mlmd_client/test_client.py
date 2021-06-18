"""Tests for tfx_addons.mlmd_client.client."""
import os

from ml_metadata.proto import metadata_store_pb2
from tfx_addons.mlmd_client import client
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.orchestration.pipeline import Pipeline


@component
def PrintComponent(word: Parameter[str]):
  print(word)


def _create_pipeline(root_dir: str):
  comp = PrintComponent(word="test")
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
  assert type(p) == client.PipelineContext


# class MetadataClientTestCase(tf.test.TestCase):
#   """MetadataClient tests
#   """

#   def setUp(self):
#     pipeline = _create_pipeline(self.get_temp_dir())
#     self.connection_config = pipeline.metadata_connection_config
#     self.pipeline_name = pipeline.pipeline_info.pipeline_name
#     self.pipeline = pipeline
#     super().setUp()

#   def test_pipeline_run_exists(self):
#     BeamDagRunner().run(self.pipeline)
#     runs = client.MetadataClient.from_pipeline(self.pipeline).runs
#     self.assertEqual(len(runs), 1)

#   def test_get_status(self):
#     BeamDagRunner().run(self.pipeline)
#     run = (client.MetadataClient(self.connection_config).get_pipeline(
#       self.pipeline_name).runs[0])
#     self.assertDictEqual(run.get_status(), {"CsvExampleGen": "complete"})

#   def test_pipeline_context_attributes(self):
#     BeamDagRunner().run(self.pipeline)
#     pipeline = client.MetadataClient.from_pipeline(self.pipeline)
#     self.assertEqual(pipeline.name, self.pipeline_name)
#     self.assertEqual(pipeline.runs[0].pipeline_name, self.pipeline_name)

#   def test_get_pipeline_not_run(self):
#     pipeline = client.MetadataClient.from_pipeline(self.pipeline)
#     self.assertIsNone(pipeline)

# if __name__ == '__main__':
#   tf.test.main()
