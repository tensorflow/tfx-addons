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
"""Tests for utils.py"""

import pathlib
from unittest import mock

import tensorflow as tf
import tensorflow_transform as tft
from absl.testing import absltest, parameterized
from ml_metadata.proto import metadata_store_pb2
from tfx import types

from tfx_addons.predictions_to_bigquery import utils


def _make_artifact(uri) -> types.Artifact:
  artifact = types.Artifact(metadata_store_pb2.ArtifactType())
  artifact.uri = uri
  return artifact


# pylint: disable=protected-access
class UtilsTest(parameterized.TestCase):
  """Tests for utils module functions."""
  def test_get_features_from_prediction_results(self):
    test_data_dir = pathlib.Path(
        'tfx_addons/predictions_to_bigquery/testdata/sample-tfx-output')
    prediction_log_path = (test_data_dir /
                           'BulkInferrer/inference_result/7/*.gz')
    output = utils._get_feature_spec_from_prediction_results(
        str(prediction_log_path))
    self.assertIn('Culmen Depth (mm)', output)
    self.assertEqual(tf.float32, output['Culmen Depth (mm)'].dtype)

  @parameterized.named_parameters([
      ('error_no_inputs', False, False, False),
      ('schema', True, False, False),
      ('tft_output', False, True, False),
      ('prediction_log_path', False, False, True),
  ])
  def test_get_feature_spec(self, has_schema, has_tft_output,
                            has_prediction_log_path):
    mock_load_schema = self.enter_context(
        mock.patch.object(utils,
                          '_get_feature_spec_from_schema_file',
                          autospec=True,
                          return_value=has_schema))
    mock_raw_feature_spec = self.enter_context(
        mock.patch.object(tft.TFTransformOutput,
                          'raw_feature_spec',
                          autospec=True))
    mock_parse_features_from_prediction_results = self.enter_context(
        mock.patch.object(utils,
                          '_get_feature_spec_from_prediction_results',
                          autospec=True,
                          return_value=has_schema))

    if (has_schema is None and has_tft_output is None
        and has_prediction_log_path is None):
      with self.assertRaises(ValueError):
        _ = utils.get_feature_spec(has_schema, has_tft_output,
                                   has_prediction_log_path)
      return

    if has_schema:
      schema = [_make_artifact('schema_uri')]
      _ = utils.get_feature_spec(schema, None, None)
      mock_load_schema.assert_called_once()

    elif has_tft_output:
      tft_output = tft.TFTransformOutput('uri')
      _ = utils.get_feature_spec(None, tft_output, None)
      mock_raw_feature_spec.assert_called_once()

    else:
      prediction_log_path = 'path'
      _ = utils.get_feature_spec(None, None, prediction_log_path)
      mock_parse_features_from_prediction_results.assert_called_once()

  @parameterized.named_parameters([
      ('no_label_field', False),
      ('with_label_field', True),
  ])
  def test_feature_spec_to_bq_schema(self, add_label_field):
    feature_spec: utils.FeatureSpec = {
        'Some Feature': tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    required = True
    if add_label_field:
      expected = {
          'fields': [
              {
                  'name': 'Some_Feature',
                  'type': 'INTEGER',
                  'mode': 'REQUIRED',
              },
              {
                  'name': 'category_label',
                  'type': 'STRING',
                  'mode': 'REQUIRED',
              },
              {
                  'name': 'score',
                  'type': 'FLOAT',
                  'mode': 'REQUIRED',
              },
              {
                  'name': 'datetime',
                  'type': 'TIMESTAMP',
                  'mode': 'REQUIRED',
              },
          ]
      }
    else:
      expected = {
          'fields': [
              {
                  'name': 'Some_Feature',
                  'type': 'INTEGER',
                  'mode': 'REQUIRED',
              },
              {
                  'name': 'score',
                  'type': 'FLOAT',
                  'mode': 'REQUIRED',
              },
              {
                  'name': 'datetime',
                  'type': 'TIMESTAMP',
                  'mode': 'REQUIRED',
              },
          ]
      }

    output = utils.feature_spec_to_bq_schema(feature_spec,
                                             required,
                                             add_label_field=add_label_field)

    self.assertEqual(expected, output)


if __name__ == '__main__':
  absltest.main()
