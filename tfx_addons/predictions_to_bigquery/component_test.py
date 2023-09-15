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
"""Tests for component.py."""

import unittest

from tfx.types import channel_utils, standard_artifacts

from tfx_addons.predictions_to_bigquery import component


class ComponentTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    self._transform_graph = channel_utils.as_channel(
        [standard_artifacts.TransformGraph()])
    self._inference_results = channel_utils.as_channel(
        [standard_artifacts.InferenceResult()])
    self._schema = channel_utils.as_channel([standard_artifacts.Schema()])

  def testInit(self):
    component_instance = component.PredictionsToBigQuery(
        transform_graph=self._transform_graph,
        inference_results=self._inference_results,
        schema=self._schema,
        bq_table_name='gcp_project:bq_database.table',
        gcs_temp_dir='gs://bucket/temp-dir',
        vocab_label_file='vocab_txt',
        filter_threshold=0.1,
        table_partitioning=False,
        table_time_suffix='%Y%m%d',
    )
    self.assertCountEqual({
        'inference_results',
        'schema',
        'transform_graph',
    }, component_instance.inputs.keys())
    self.assertCountEqual({'bigquery_export'},
                          component_instance.outputs.keys())
    self.assertCountEqual(
        {
            'bq_table_name',
            'gcs_temp_dir',
            'table_expiration_days',
            'filter_threshold',
            'table_partitioning',
            'table_time_suffix',
            'vocab_label_file',
        }, component_instance.exec_properties.keys())


if __name__ == '__main__':
  unittest.main()
