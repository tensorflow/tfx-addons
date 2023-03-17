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
"""Integration test for the predictions-to-bigquery component.

Prerequisites:
- 'GOOGLE_CLOUD_PROJECT' environmental variable must be set containing
  the GCP project ID to be used for testing.
- 'GCS_TEMP_DIR' environmental variable must be set containing the
  Cloud Storage URI to use for handling temporary files as part of the
  BigQuery export process. e.g. `gs://path/to/temp/dir`.
- BigQuery API must be enabled on the Cloud project.
"""

import datetime
import json
import logging
import os
import pathlib
import shutil

from absl.testing import absltest
from google.api_core import exceptions
from google.cloud import bigquery
from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx import v1 as tfx
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types.standard_artifacts import Model, String

from tfx_addons.predictions_to_bigquery import component, executor

_GOOGLE_CLOUD_PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
_GCS_TEMP_DIR = os.environ['GCS_TEMP_DIR']

_BQ_TABLE_EXPIRATION_DATE = datetime.datetime.now() + datetime.timedelta(
    days=1)

_TEST_DATA_DIR = pathlib.Path('tfx_addons/predictions_to_bigquery/testdata')


def _make_artifact(uri: pathlib.Path) -> types.Artifact:
  artifact = types.Artifact(metadata_store_pb2.ArtifactType())
  artifact.uri = str(uri)
  return artifact


def _make_artifact_mapping(
    data_dict: dict[str, pathlib.Path]) -> dict[str, list[types.Artifact]]:
  return {k: [_make_artifact(v)] for k, v in data_dict.items()}


@absltest.skip
class ExecutorBigQueryTest(absltest.TestCase):
  """Tests executor pipeline exporting predicitons to a BigQuery table."""
  def _get_full_bq_table_name(self, generated_bq_table_name):
    return f'{self.gcp_project}.{self.bq_dataset}.{generated_bq_table_name}'

  def _assert_bq_table_exists(self, full_bq_table_name):
    full_bq_table_name = full_bq_table_name.replace(':', '.')
    try:
      self.client.get_table(full_bq_table_name)
    except exceptions.NotFound as e:
      self.fail(f'BigQuery table not found: {full_bq_table_name} . '
                f'Reason: {e} .')

  def _expire_table(self, full_bq_table_name):
    try:
      table = self.client.get_table(full_bq_table_name)
    except (ValueError, exceptions.NotFound):
      logging.warning('Unable to read table: %s', full_bq_table_name)
    else:
      table.expires = _BQ_TABLE_EXPIRATION_DATE
      self.client.update_table(table, ['expires'])

  def setUp(self):
    super().setUp()
    self.test_data_dir = _TEST_DATA_DIR / 'sample-tfx-output'
    self.input_dict = _make_artifact_mapping({
        'transform_graph':
        (self.test_data_dir / 'Transform/transform_graph/5'),
        'inference_results':
        (self.test_data_dir / 'BulkInferrer/inference_result/7'),
        'schema':
        (self.test_data_dir / 'Transform/transform_graph/5/metadata'),
    })
    self.temp_dir = self.create_tempdir()
    self.output_dict = _make_artifact_mapping(
        {'bigquery_export': pathlib.Path(self.temp_dir.full_path)})
    self.gcp_project = _GOOGLE_CLOUD_PROJECT
    self.bq_dataset = 'executor_bigquery_test_dataset'
    self.bq_table_name = f'{self.gcp_project}:{self.bq_dataset}.predictions'
    self.client = bigquery.Client()
    self.client.create_dataset(dataset=self.bq_dataset, exists_ok=True)
    self.exec_properties = {
        'bq_table_name': self.bq_table_name,
        'table_expiration_days': 5,
        'filter_threshold': 0.5,
        'gcs_temp_dir': _GCS_TEMP_DIR,
        'table_partitioning': False,
        'table_time_suffix': '%Y%m%d%H%M%S',
        'vocab_label_file': 'Species',
    }
    self.generated_bq_table_name = None

    self.executor = executor.Executor()

  def tearDown(self):
    super().tearDown()
    self._expire_table(self.generated_bq_table_name)

  def test_Do(self):
    self.executor.Do(self.input_dict, self.output_dict, self.exec_properties)
    self.assertIsNotNone(self.output_dict['bigquery_export'])
    bigquery_export = artifact_utils.get_single_instance(
        self.output_dict['bigquery_export'])
    self.generated_bq_table_name = (
        bigquery_export.get_custom_property('generated_bq_table_name'))
    # Expected table name format by BigQuery client: project.dataset.table_name
    self.generated_bq_table_name = (str(self.generated_bq_table_name).replace(
        ':', '.'))
    self._assert_bq_table_exists(self.generated_bq_table_name)


@tfx.dsl.components.component
def _saved_model_component(
    model: tfx.dsl.components.OutputArtifact[Model],
    saved_model_dir: tfx.dsl.components.Parameter[str],
):
  """Creates a component that outputs a TF saved model."""
  target_dir = os.path.join(model.uri, 'Format-Serving')
  os.makedirs(target_dir, exist_ok=True)
  shutil.copytree(saved_model_dir, target_dir, dirs_exist_ok=True)


@tfx.dsl.components.component
def _get_predictions_to_bigquery_output(
    bigquery_export: tfx.dsl.components.InputArtifact[String],
    output_filepath: tfx.dsl.components.Parameter[str],
):
  """Checks output of the predictions-to-bigquery component."""
  generated_bq_table_name = bigquery_export.get_custom_property(
      'generated_bq_table_name')
  output = {
      'generated_bq_table_name': generated_bq_table_name,
  }
  with open(output_filepath, 'wt', encoding='utf-8') as output_file:
    json.dump(output, output_file)


class ComponentIntegrationTest(absltest.TestCase):
  """Tests component integration with other TFX components/services."""
  def setUp(self):
    super().setUp()
    # Pipeline config
    self.dataset_dir = _TEST_DATA_DIR / 'penguins-dataset'
    self.saved_model_dir = (_TEST_DATA_DIR /
                            'sample-tfx-output/Trainer/model/6/Format-Serving')
    self.model_channel = types.Channel(type=Model)
    self.pipeline_name = 'component_integration_test'
    self.pipeline_root = self.create_tempdir()
    self.metadata_path = self.create_tempfile()
    self.gcs_temp_dir = _GCS_TEMP_DIR
    self.output_file = self.create_tempfile()

    # GCP config
    self.gcp_project = _GOOGLE_CLOUD_PROJECT
    self.bq_dataset = 'component_integration_test_dataset'
    self.bq_table_name = f'{self.gcp_project}:{self.bq_dataset}.predictions'
    self.client = bigquery.Client()
    self.client.create_dataset(dataset=self.bq_dataset, exists_ok=True)

    # Components
    test_split = (example_gen_pb2.Input.Split(name='test',
                                              pattern='test/test-tiny.csv'))
    self.unlabeled_example_gen = tfx.components.CsvExampleGen(
        input_base=str(self.dataset_dir),
        input_config=example_gen_pb2.Input(
            splits=[test_split])).with_id('UnlabeledExampleGen')
    self.saved_model = _saved_model_component(saved_model_dir=str(
        self.saved_model_dir))  # type: ignore
    self.bulk_inferrer = tfx.components.BulkInferrer(
        examples=self.unlabeled_example_gen.outputs['examples'],
        model=self.saved_model.outputs['model'],
        data_spec=tfx.proto.DataSpec(),
        model_spec=tfx.proto.ModelSpec(),
    )

    # Test config
    self.generated_bq_table_name = None

  def tearDown(self):
    super().tearDown()
    self._expire_table(self.generated_bq_table_name)

  def _expire_table(self, full_bq_table_name):
    full_bq_table_name = full_bq_table_name.replace(':', '.')
    try:
      table = self.client.get_table(full_bq_table_name)
    except (ValueError, exceptions.NotFound):
      logging.warning('Unable to read table: %s', full_bq_table_name)
    else:
      table.expires = _BQ_TABLE_EXPIRATION_DATE
      self.client.update_table(table, ['expires'])

  def _create_pipeline(self, component_under_test, output_filepath):
    get_output = (_get_predictions_to_bigquery_output(
        bigquery_export=component_under_test.outputs['bigquery_export'],
        output_filepath=output_filepath))
    components = [
        self.unlabeled_example_gen,
        self.saved_model,
        self.bulk_inferrer,
        component_under_test,
        get_output,
    ]
    return tfx.dsl.Pipeline(
        pipeline_name=self.pipeline_name,
        pipeline_root=str(self.pipeline_root.full_path),
        metadata_connection_config=(
            tfx.orchestration.metadata.sqlite_metadata_connection_config(
                self.metadata_path.full_path)),
        components=components)

  def _run_pipeline(self, component_under_test):
    output_tempfile = self.create_tempfile()
    pipeline = self._create_pipeline(component_under_test,
                                     output_tempfile.full_path)
    tfx.orchestration.LocalDagRunner().run(pipeline)
    with open(output_tempfile.full_path, encoding='utf-8') as output_file:
      output = json.load(output_file)
    return output

  def test_bulk_inferrer_bigquery_integration(self):
    """Tests component integration with BulkInferrer and BigQuery."""
    predictions_to_bigquery = component.PredictionsToBigQueryComponent(
        inference_results=self.bulk_inferrer.outputs['inference_result'],
        bq_table_name=self.bq_table_name,
        gcs_temp_dir=self.gcs_temp_dir,
    )

    output = self._run_pipeline(predictions_to_bigquery)
    self.generated_bq_table_name = output['generated_bq_table_name']
    self.assertStartsWith(self.generated_bq_table_name, self.bq_table_name)


if __name__ == '__main__':
  absltest.main()
