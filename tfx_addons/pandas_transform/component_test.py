# Copyright 2022 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx_addons.pandas_transform.component."""

import apache_beam as beam
import tensorflow as tf
from packaging.version import Version, parse
from tfx import v1 as tfx
from tfx.orchestration import data_types
from tfx.types import (artifact_utils, channel_utils, standard_artifacts,
                       standard_component_specs)

from tfx_addons.pandas_transform import component


class ComponentTest(tf.test.TestCase):
  def setUp(self):
    super().setUp()
    examples_artifact = standard_artifacts.Examples()
    examples_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    self.examples = channel_utils.as_channel([examples_artifact])
    self.schema = channel_utils.as_channel([standard_artifacts.Schema()])
    statistics_artifact = standard_artifacts.ExampleStatistics()
    statistics_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    self.statistics = channel_utils.as_channel([statistics_artifact])

  def _verify_outputs(self, pandas_transform):
    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME, pandas_transform.outputs[
            standard_component_specs.TRANSFORMED_EXAMPLES_KEY].type_name)

  def test_construct_from_module_file(self):
    module_file = './null_preprocessing.py'
    if parse(tfx.__version__) >= Version('1.8.0'):
      pandas_transform = component.PandasTransform(
          examples=self.examples,
          schema=self.schema,
          statistics=self.statistics,
          module_file=module_file,
          beam_pipeline=beam.Pipeline())
    else:
      pandas_transform = component.PandasTransform(examples=self.examples,
                                                   schema=self.schema,
                                                   statistics=self.statistics,
                                                   module_file=module_file)
    self._verify_outputs(pandas_transform)
    self.assertEqual(
        module_file, pandas_transform.exec_properties[
            standard_component_specs.MODULE_FILE_KEY])

  def test_construct_with_parameter(self):
    module_file = data_types.RuntimeParameter(name='module-file', ptype=str)
    if parse(tfx.__version__) >= Version('1.8.0'):
      pandas_transform = component.PandasTransform(
          examples=self.examples,
          schema=self.schema,
          statistics=self.statistics,
          module_file=module_file,
          beam_pipeline=beam.Pipeline())
    else:
      pandas_transform = component.PandasTransform(examples=self.examples,
                                                   schema=self.schema,
                                                   statistics=self.statistics,
                                                   module_file=module_file)
    self._verify_outputs(pandas_transform)
    self.assertJsonEqual(
        str(module_file),
        str(pandas_transform.exec_properties[
            standard_component_specs.MODULE_FILE_KEY]))


#   NOTE: This test is not currently working because
#         Python-function components don't currently
#         propagate exceptions correctly.  b/238368874
#
#   def test_construct_missing_user_module(self):
#     module_file = './missing_file.py'
#     with self.assertRaises(ImportError):
#         pandas_transform = component.PandasTransform(
#           examples=self.examples,
#           schema=self.schema,
#           statistics=self.statistics,
#           module_file=module_file
#       )

if __name__ == '__main__':
  tf.test.main()
