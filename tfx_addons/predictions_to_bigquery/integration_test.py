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
"""Integration test for the predictions-to-bigquery component."""

import datetime
import logging
import os
import pathlib

from absl.testing import absltest
from google.api_core import exceptions
from google.cloud import bigquery
from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.types import artifact_utils

from tfx_addons.predictions_to_bigquery import executor

_BQ_TABLE_EXPIRATION_DATE = datetime.datetime.now() + datetime.timedelta(
    days=1)


def _make_artifact(uri: pathlib.Path) -> types.Artifact:
  artifact = types.Artifact(metadata_store_pb2.ArtifactType())
  artifact.uri = str(uri)
  return artifact


def _make_artifact_mapping(
    data_dict: dict[str, pathlib.Path]) -> dict[str, list[types.Artifact]]:
  return {k: [_make_artifact(v)] for k, v in data_dict.items()}


class ExecutorBigQueryTest(absltest.TestCase):
  """Tests executor pipeline exporting predicitons to a BigQuery table.

  Prerequisites:
    - 'GOOGLE_CLOUD_PROJECT' environmental variable must be set.
    - BigQuery API must be enabled.
    - A BigQuery dataset named 'test_dataset' should exist.
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
    self.test_data_dir = pathlib.Path(
        'tfx_addons/predictions_to_bigquery/testdata/sample-tfx-output')
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
    self.gcp_project = os.environ['GOOGLE_CLOUD_PROJECT']
    self.bq_dataset = 'executor_bigquery_test_dataset'
    self.bq_table_name = f'{self.gcp_project}:{self.bq_dataset}.predictions'
    self.client = bigquery.Client()
    self.client.create_dataset(dataset=self.bq_dataset, exists_ok=True)
    self.exec_properties = {
        'bq_table_name': self.bq_table_name,
        'table_expiration_days': 5,
        'filter_threshold': 0.5,
        'gcs_temp_dir': 'gs://pred2bq-bucket/temp-dir',
        'table_partitioning': False,
        'table_time_suffix': '%Y%m%d%H%M%S',
        'vocab_label_file': 'Species',
    }
    self.generated_bq_table_name = None

    self.executor = executor.Executor()

  def tearDown(self):
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


if __name__ == '__main__':
  absltest.main()
