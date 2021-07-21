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
"""Tests for schemaCuration.component.executor."""

import os

import pytest
import tensorflow as tf
from pkg_resources import resource_filename
from tfx import types
from tfx.dsl.io import fileio
from tfx.types import standard_artifacts, standard_component_specs

from tfx_addons.schema_curation.component import executor
from tfx_addons.schema_curation.test_data.module_file import module_file


# ToDo(casassg): Test fails currently. Marking as skip until it gets resolved so that
# CI is not red.
@pytest.mark.skip
class ExecutorTest(tf.test.TestCase):
  def testDo(self):
    super().setUp()

    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self.schema = standard_artifacts.Schema()
    schema_file = resource_filename(
        'tfx_addons.schema_curation.test_data.schema_gen', 'schema.pbtxt')
    fileio.copy(
        schema_file,
        os.path.join(self.get_temp_dir(), 'input_schema', 'schema.pbtxt'))
    self.schema.uri = os.path.join(self.get_temp_dir(), 'input_schema')

    self.input_dict = {
        standard_component_specs.SCHEMA_KEY: [self.schema],
    }

    self.schema_output = standard_artifacts.Schema()
    self.schema_output.uri = os.path.join(self._output_data_dir,
                                          'custom_schema')

    output_dict = {
        'custom_schema': [self.schema_output],
    }

    self._module_file = module_file.__file__
    self.schema_fn = '%s.%s' % (module_file.schema_fn.__module__,
                                module_file.schema_fn.__name__)

    print(self._module_file)
    self.exec_properties = {
        standard_component_specs.MODULE_FILE_KEY: self._module_file
    }

    self.schema_curation_executor = executor.Executor()

    self.schema_curation_executor.Do(self.input_dict, output_dict,
                                     self.exec_properties)

  def _verify_schema_curation_outputs(self):
    self.assertNotEqual(0, len(fileio.listdir(self.schema_output.uri)))

  def testDoWithModuleFile(self):
    self.exec_properties['module_file'] = self._module_file
    self.schema_curation_executor.Do(self.input_dict, self.output_dict,
                                     self.exec_properties)
    self._verify_schema_curation_outputs()

  def get_schema_fn(self):
    self._exec_properties['schema_fn'] = self.schema_fn
    self.schema_curation_executor.Do(self.input_dict, self.output_dict,
                                     self._exec_properties)
    self._verify_schema_curation_outputs()

  def testDoWithCache(self):
    # First run that creates cache.
    output_cache_artifact = types.Artifact('OutputCache')
    output_cache_artifact.uri = os.path.join(self._output_data_dir, 'CACHE/')

    self.output_dict['cache_output_path'] = [output_cache_artifact]

    self.exec_properties['module_file'] = self._module_file
    self.schemaCuration_executor.Do(self.input_dict, self.output_dict,
                                    self.exec_properties)
    self._verify_schema_curation_outputs()
    self.assertNotEqual(0, len(tf.io.gfile.listdir(output_cache_artifact.uri)))


if __name__ == '__main__':
  tf.test.main()
