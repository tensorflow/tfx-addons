# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for model_card_toolkit.tfx.component."""

import json as json_lib

from absl.testing import absltest
from tfx.types import channel_utils, standard_artifacts

from tfx_addons.model_card_generator import artifact
from tfx_addons.model_card_generator.component import ModelCardGenerator


class ComponentTest(absltest.TestCase):
  def test_component_construction(self):
    this_component = ModelCardGenerator(
        statistics=channel_utils.as_channel(
            [standard_artifacts.ExampleStatistics()]),
        evaluation=channel_utils.as_channel(
            [standard_artifacts.ModelEvaluation()]),
        pushed_model=channel_utils.as_channel(
            [standard_artifacts.PushedModel()]),
        json=json_lib.dumps(
            {'model_details': {
                'name': 'my model',
                'version': {
                    'name': 'v1'
                }
            }}),
        template_io=[('path/to/html/template', 'mc.html'),
                     ('path/to/md/template', 'mc.md')])

    with self.subTest('outputs'):
      self.assertEqual(this_component.outputs['model_card'].type_name,
                       artifact.ModelCard.TYPE_NAME)

    with self.subTest('exec_properties'):
      self.assertDictEqual(
          {
              'json':
              json_lib.dumps({
                  'model_details': {
                      'name': 'my model',
                      'version': {
                          'name': 'v1'
                      }
                  }
              }),
              'template_io': [('path/to/html/template', 'mc.html'),
                              ('path/to/md/template', 'mc.md')]
          }, this_component.exec_properties)

  def test_empty_component_construction(self):
    this_component = ModelCardGenerator()
    with self.subTest('outputs'):
      self.assertEqual(this_component.outputs['model_card'].type_name,
                       artifact.ModelCard.TYPE_NAME)


if __name__ == '__main__':
  absltest.main()
