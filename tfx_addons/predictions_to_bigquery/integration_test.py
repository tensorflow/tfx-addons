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
"""Integration test for PredictionsToBigQuery component.

Prerequisites:

The following environmental variables should be defined.

  GOOGLE_CLOUD_PROJECT: environmental variable must be set containing
    the GCP project ID to be used for testing.

  GCS_TEMP_DIR: Cloud Storage URI to use for handling temporary files as part
    of the BigQuery export process. e.g. `gs://path/to/temp/dir`.

  GCP_SERVICE_ACCOUNT_EMAIL: Service account address to use for Vertex AI
    pipeline runs. The service account should be have access to Cloud
    Storage and Vertex AI. Local test runs may still work without this variable.

  GCP_COMPONENT_IMAGE: Docker image repository name that would be used for
    Vertex AI Pipelines integration testing. The Dockerfile associated with
    this component will create a custom TFX image with the component that
    should be uploaded to Artifact Registry.
    A new Docker image should be uploaded whenever there are any changes
    to the non-test module files of this component.


The following Google Cloud APIs should be enabled

  BigQuery API: For generating the BigQuery table output of this component.

  Vertex AI API: For running TFX pipeline jobs in Vertex.

  Artifact Registry API: For storing the Docker image to be used in order
    to run a TFX pipeline with this component in Vertex AI.

Vertex AI test:

The `ComponentIntegrationTest` test class has a test to run the component
in Vertex AI Pipelines. The test is skipped by default, since it can take
several minutes to complete. You can comment out the skip decorator
(i.e. `@absltest.skip(...)`) and add similar decorators to other tests that
you don't want to run.
"""

import datetime
import json
import logging
import os
import pathlib
import shutil
import subprocess
from typing import List

import tensorflow as tf
from absl.testing import absltest, parameterized
from google.api_core import exceptions
from google.cloud import aiplatform, bigquery
from google.cloud.aiplatform import pipeline_jobs
from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx import v1 as tfx
from tfx.dsl.component.experimental import container_component, placeholders
from tfx.dsl.components.base import base_node
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types.standard_artifacts import Model, String, TransformGraph

from tfx_addons.predictions_to_bigquery import component, executor

_GOOGLE_CLOUD_PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
_GCS_TEMP_DIR = os.environ['GCS_TEMP_DIR']
_GCP_SERVICE_ACCOUNT_EMAIL = os.environ.get('GCP_SERVICE_ACCOUNT_EMAIL')
_GCP_COMPONENT_IMAGE = os.environ['GCP_COMPONENT_IMAGE']

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


class ExecutorBigQueryTest(absltest.TestCase):
  """Tests executor pipeline exporting predictions to a BigQuery table.

  This test generates a BigQuery table with an expiration date of 1 day.
  """
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
    self.temp_file = self.create_tempfile()
    self.output_dict = _make_artifact_mapping(
        {'bigquery_export': pathlib.Path(self.temp_file.full_path)})
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
    if self.generated_bq_table_name:
      self._expire_table(self.generated_bq_table_name)

  def test_Do(self):
    self.executor.Do(self.input_dict, self.output_dict, self.exec_properties)
    self.assertIsNotNone(self.output_dict['bigquery_export'])
    bigquery_export = artifact_utils.get_single_instance(
        self.output_dict['bigquery_export'])
    self.generated_bq_table_name = (
        bigquery_export.get_custom_property('generated_bq_table_name'))
    # Expected table name format by BigQuery client: project.dataset.table_name
    with open(self.temp_file.full_path, encoding='utf-8') as input_file:
      self.generated_bq_table_name = input_file.read()
    self.generated_bq_table_name = (str(self.generated_bq_table_name).replace(
        ':', '.'))
    self._assert_bq_table_exists(self.generated_bq_table_name)


def _gcs_path_exists(gcs_path: str) -> bool:
  files = tf.io.gfile.glob(gcs_path + '/*')
  return bool(files)


def _copy_local_dir_to_gcs(local_dir: str, gcs_path: str):
  subprocess.check_call(f'gsutil -m cp -r {local_dir} {gcs_path}', shell=True)


@tfx.dsl.components.component
def _transform_function_component(
    transform_graph: tfx.dsl.components.OutputArtifact[TransformGraph],
    transform_dir: tfx.dsl.components.Parameter[str],
):
  """TFX Transform component stub."""
  os.makedirs(transform_graph.uri, exist_ok=True)
  shutil.copytree(transform_dir, transform_graph.uri, dirs_exist_ok=True)


def _create_transform_container_component_class():
  return container_component.create_container_component(
      name='TransformContainerComponent',
      inputs={},
      outputs={
          'transform_graph': TransformGraph,
      },
      parameters={
          'transform_dir': str,
      },
      image='google/cloud-sdk:latest',
      command=[
          'sh',
          '-exc',
          '''
          transform_dir="$0"
          transform_graph_uri="$1"
          gsutil cp -r $transform_dir/* $transform_graph_uri/
          ''',
          placeholders.InputValuePlaceholder('transform_dir'),
          placeholders.OutputUriPlaceholder('transform_graph'),
      ],
  )


def _transform_component(transform_dir: str):
  if transform_dir.startswith('gs://'):
    transform_class = _create_transform_container_component_class()
    transform = transform_class(transform_dir=transform_dir)
  else:
    transform = _transform_function_component(transform_dir=transform_dir)
  return transform


@tfx.dsl.components.component
def _saved_model_function_component(
    model: tfx.dsl.components.OutputArtifact[Model],
    saved_model_dir: tfx.dsl.components.Parameter[str],
):
  """Creates a component that outputs a TF saved model."""
  target_dir = os.path.join(model.uri, 'Format-Serving')
  os.makedirs(target_dir, exist_ok=True)
  shutil.copytree(saved_model_dir, target_dir, dirs_exist_ok=True)


def _create_saved_model_container_component_class():
  return container_component.create_container_component(
      name='SavedModelContainerComponent',
      inputs={},
      outputs={
          'model': Model,
      },
      parameters={
          'saved_model_dir': str,
      },
      image='google/cloud-sdk:latest',
      command=[
          'sh',
          '-exc',
          '''
          saved_model_dir="$0"
          model_uri="$1"
          gsutil cp -r $saved_model_dir $model_uri/
          ''',
          placeholders.InputValuePlaceholder('saved_model_dir'),
          placeholders.OutputUriPlaceholder('model'),
      ],
  )


def _saved_model_component(saved_model_dir: str):
  if saved_model_dir.startswith('gs://'):
    saved_model_component_class = (
        _create_saved_model_container_component_class())
    saved_model = saved_model_component_class(saved_model_dir=saved_model_dir)
  else:
    saved_model = _saved_model_function_component(
        saved_model_dir=saved_model_dir)
  return saved_model


@tfx.dsl.components.component
def _get_output_function_component(
    bigquery_export: tfx.dsl.components.InputArtifact[String],
    output_filepath: tfx.dsl.components.Parameter[str],
):
  """Copies component-under-test output to `output_filepath`."""
  with tf.io.gfile.GFile(bigquery_export.uri) as input_file:
    bq_table_name = input_file.read()
  with tf.io.gfile.GFile(output_filepath, 'w') as output_file:
    output = {
        'generated_bq_table_name': bq_table_name,
    }
    json.dump(output, output_file)


def _create_get_output_container_component_class():
  return container_component.create_container_component(
      name='BigQueryExportContainerComponent',
      inputs={
          'bigquery_export': String,
      },
      parameters={
          'output_path': str,
      },
      image='google/cloud-sdk:latest',
      command=[
          'sh',
          '-exc',
          '''
          apt install -y jq
          bigquery_export_uri="$0"
          local_bigquery_export_path=$(mktemp)
          local_output_path=$(mktemp)
          output_path="$1"
          gsutil cp $bigquery_export_uri $local_bigquery_export_path
          bq_table_name=$(cat $local_bigquery_export_path)
          jq --null-input \
              --arg bq_table_name "$bq_table_name" \
              '{"generated_bq_table_name": $bq_table_name}' \
              > $local_output_path
          gsutil cp -r $local_output_path $output_path
          ''',
          placeholders.InputUriPlaceholder('bigquery_export'),
          placeholders.InputValuePlaceholder('output_path'),
      ],
  )


def _get_output_component(output_channel, output_file):
  if output_file.startswith('gs://'):
    get_output_class = _create_get_output_container_component_class()
    output_component = get_output_class(bigquery_export=output_channel,
                                        output_path=output_file)
  else:
    output_component = _get_output_function_component(
        bigquery_export=output_channel, output_filepath=output_file)
  return output_component


class ComponentIntegrationTest(parameterized.TestCase):
  """Tests component integration with other TFX components/services.

  This test generates a BigQuery table with an expiration date of 1 day.
  """
  def setUp(self):
    super().setUp()
    # Pipeline config
    self.pipeline_name = 'component-integration-test'
    self.test_file = 'test-tiny.csv'
    self.gcs_temp_dir = _GCS_TEMP_DIR
    self.dataset_name = 'penguins-dataset'
    self.saved_model_path = 'sample-tfx-output/Trainer/model/6/Format-Serving'
    self.transform_path = 'sample-tfx-output/Transform/transform_graph/5'

    # Vertex Pipeline config
    self.service_account = _GCP_SERVICE_ACCOUNT_EMAIL
    self.location = os.environ.get('GCP_REGION') or 'us-central1'

    # GCP config
    self.gcp_project = _GOOGLE_CLOUD_PROJECT
    self.bq_dataset = 'component_integration_test_dataset'
    self.bq_table_name = f'{self.gcp_project}:{self.bq_dataset}.predictions'
    self.client = bigquery.Client()
    self.client.create_dataset(dataset=self.bq_dataset, exists_ok=True)

    # Test config
    self.generated_bq_table_name = None

  def tearDown(self):
    super().tearDown()
    if self.generated_bq_table_name is not None:
      self._expire_table(self.generated_bq_table_name)

  def _add_test_label_to_table(self, table):
    labels = {'test_method_name': self._testMethodName}
    table.labels = labels
    self.client.update_table(table, ['labels'])

  def _expire_table(self, full_bq_table_name):
    full_bq_table_name = full_bq_table_name.replace(':', '.')
    try:
      table = self.client.get_table(full_bq_table_name)
    except (ValueError, exceptions.NotFound):
      logging.warning('Unable to read table: %s', full_bq_table_name)
    else:
      table.expires = _BQ_TABLE_EXPIRATION_DATE
      table = self.client.update_table(table, ['expires'])
      self._add_test_label_to_table(table)

  def _create_gcs_tempfile(self) -> str:
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    self.gcs_temp_file = os.path.join(_GCS_TEMP_DIR, 'pipeline-outputs',
                                      f'output-{timestamp}')
    return self.gcs_temp_file

  def _create_upstream_component_map(self, use_gcs=False):
    if use_gcs:
      gcs_test_data_dir = os.path.join(_GCS_TEMP_DIR, _TEST_DATA_DIR.stem)
      if not _gcs_path_exists(gcs_test_data_dir):
        # Copy test files to GCS
        # NOTE: If local `testdata` files are updated, forcing a copy to the
        # GCS mirror directory may be needed.
        _copy_local_dir_to_gcs(str(_TEST_DATA_DIR), _GCS_TEMP_DIR)
      dataset_dir = os.path.join(gcs_test_data_dir, self.dataset_name)
      saved_model_dir = os.path.join(gcs_test_data_dir, self.saved_model_path)
      transform_dir = os.path.join(gcs_test_data_dir, self.transform_path)
    else:
      dataset_dir = str(_TEST_DATA_DIR / self.dataset_name)
      saved_model_dir = str(_TEST_DATA_DIR / self.saved_model_path)
      transform_dir = str(_TEST_DATA_DIR / self.transform_path)

    test_split = example_gen_pb2.Input.Split(name='test',
                                             pattern=f'test/{self.test_file}')
    example_gen = tfx.components.CsvExampleGen(
        input_base=dataset_dir,
        input_config=example_gen_pb2.Input(
            splits=[test_split])).with_id('UnlabeledExampleGen')

    transform = _transform_component(transform_dir=transform_dir)

    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'], )

    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'], )

    saved_model = _saved_model_component(saved_model_dir)

    bulk_inferrer = tfx.components.BulkInferrer(
        examples=example_gen.outputs['examples'],
        model=saved_model.outputs['model'],
        data_spec=tfx.proto.DataSpec(),
        model_spec=tfx.proto.ModelSpec(),
    )

    return {
        'example_gen': example_gen,
        'statistics_gen': statistics_gen,
        'schema_gen': schema_gen,
        'transform': transform,
        'saved_model': saved_model,
        'bulk_inferrer': bulk_inferrer,
    }

  def _create_pipeline(self,
                       component_under_test: base_node.BaseNode,
                       upstream_components: List[base_node.BaseNode],
                       output_file: str,
                       pipeline_dir: str,
                       metadata_connection_config=None):
    output_component = _get_output_component(
        component_under_test.outputs['bigquery_export'], output_file)
    components = (upstream_components +
                  [component_under_test, output_component])
    return tfx.dsl.Pipeline(
        pipeline_name=self.pipeline_name,
        pipeline_root=pipeline_dir,
        components=components,
        metadata_connection_config=metadata_connection_config)

  def _run_local_pipeline(self, pipeline):
    assert pipeline.metadata_connection_config is not None
    return tfx.orchestration.LocalDagRunner().run(pipeline)

  def _run_vertex_pipeline(self, pipeline):
    pipeline_definition_file = os.path.join(
        '/tmp', f'{self.pipeline_name}-pipeline.json')
    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(
            default_image=_GCP_COMPONENT_IMAGE),
        output_filename=pipeline_definition_file)
    runner.run(pipeline)

    aiplatform.init(project=_GOOGLE_CLOUD_PROJECT, location=self.location)
    job = pipeline_jobs.PipelineJob(template_path=pipeline_definition_file,
                                    display_name=self.pipeline_name)
    return job.run(service_account=self.service_account, sync=True)

  def _check_output(self, output_file: str):
    with tf.io.gfile.GFile(output_file) as output_file_handler:
      output = json.load(output_file_handler)

    self.generated_bq_table_name = output['generated_bq_table_name']
    self.assertStartsWith(self.generated_bq_table_name, self.bq_table_name)

  @parameterized.named_parameters([
      (
          'inference_results_only',
          False,
          False,
      ),
      ('inference_results_schema', True, False),
      ('inference_results_transform', False, True),
      ('inference_results_schema_transform', True, True),
  ])
  def test_local_pipeline(self, add_schema, add_transform):
    """Tests component using a local pipeline runner."""
    upstream = self._create_upstream_component_map()
    upstream_components = [
        upstream['example_gen'],
        upstream['saved_model'],
        upstream['bulk_inferrer'],
    ]

    if add_schema:
      upstream_components.append(upstream['statistics_gen'])
      upstream_components.append(upstream['schema_gen'])
      schema = upstream['schema_gen'].outputs['schema']
    else:
      schema = None

    if add_transform:
      transform_graph = upstream['transform'].outputs['transform_graph']
      upstream_components.append(upstream['transform'])
      vocab_label_file = 'Species'
    else:
      transform_graph = None
      vocab_label_file = None

    component_under_test = component.PredictionsToBigQuery(
        inference_results=(
            upstream['bulk_inferrer'].outputs['inference_result']),
        transform_graph=transform_graph,
        schema=schema,
        bq_table_name=self.bq_table_name,
        gcs_temp_dir=self.gcs_temp_dir,
        vocab_label_file=vocab_label_file,
    )

    output_file = self.create_tempfile()
    pipeline_dir = self.create_tempdir()
    metadata_path = self.create_tempfile()
    metadata_connection_config = (
        tfx.orchestration.metadata.sqlite_metadata_connection_config(
            metadata_path.full_path))

    pipeline = self._create_pipeline(
        component_under_test,
        upstream_components,
        output_file.full_path,
        pipeline_dir.full_path,
        metadata_connection_config,
    )
    self._run_local_pipeline(pipeline)

    self._check_output(output_file.full_path)

  @absltest.skip('long-running test')
  def test_vertex_pipeline(self):
    """Tests component using Vertex AI Pipelines.

    This tests the case where a Transform component is used for the input
    schema.
    """
    upstream = self._create_upstream_component_map(use_gcs=True)
    upstream_components = [
        upstream['example_gen'],
        upstream['transform'],
        upstream['saved_model'],
        upstream['bulk_inferrer'],
    ]
    transform_graph = upstream['transform'].outputs['transform_graph']
    vocab_label_file = 'Species'

    component_under_test = component.PredictionsToBigQuery(
        inference_results=(
            upstream['bulk_inferrer'].outputs['inference_result']),
        transform_graph=transform_graph,
        bq_table_name=self.bq_table_name,
        gcs_temp_dir=self.gcs_temp_dir,
        vocab_label_file=vocab_label_file,
    )

    output_file = self._create_gcs_tempfile()
    pipeline_dir = os.path.join(_GCS_TEMP_DIR, 'pipeline-root')

    pipeline = self._create_pipeline(
        component_under_test,
        upstream_components,
        output_file,
        pipeline_dir,
    )
    self._run_vertex_pipeline(pipeline)

    self._check_output(output_file)


if __name__ == '__main__':
  absltest.main()
