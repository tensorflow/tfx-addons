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
"""Tests for tfx_addons.feature_selection.component"""

import os
from tfx.orchestration import metadata
from typing import Text, Optional, List
import tfx
import tensorflow as tf

from tfx_addons.feature_selection import component

def _create_pipeline(
    pipeline_name: Text, 
    pipeline_root: Text, 
    data_root: Text, 
    module_path: Text,
    metadata_path: Text,
    beam_pipeline_args: Optional[List[Text]] = None
) -> tfx.v1.dsl.Pipeline:

    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # creating and executing the FeatureSelection artifact
    feature_selection = component.FeatureSelection(
        orig_examples=example_gen.outputs['examples'],
        module_path=module_path)

    components = [
        example_gen,
        feature_selection
    ]

    return tfx.v1.dsl.Pipeline(
        pipeline_name=pipeline_name, 
        pipeline_root=pipeline_root, 
        components=components,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
        beam_pipeline_args=beam_pipeline_args)


class FeatureSelectionTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self._test_dir = os.path.join(
            os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
            self._testMethodName)
        self._feature_selection_root = os.path.dirname(__file__)

        self._pipeline_name = 'feature_selection'
        self._data_root = os.path.join(self._feature_selection_root, 'test')
        self._module_path = os.path.join(self._feature_selection_root, 'example', 'modules', 'iris_module_file.py')
        self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                        self._pipeline_name)
        
        self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

    def assertExecutedOnce(self, component: Text) -> None:
        """Check the component is executed exactly once."""
        component_path = os.path.join(self._pipeline_root, component)
        self.assertTrue(tfx.dsl.io.fileio.exists(component_path))
        execution_path = os.path.join(component_path, '.system',
                                    'executor_execution')
        execution = tfx.dsl.io.fileio.listdir(execution_path)
        self.assertLen(execution, 1)


    def assertPipelineExecution(self) -> None:
        self.assertExecutedOnce('CsvExampleGen')
        self.assertExecutedOnce('FeatureSelection')


    def testFeatureSelectionPipelineLocal(self):
        tfx.v1.orchestration.LocalDagRunner().run(
            _create_pipeline(
                pipeline_name=self._pipeline_name,
                pipeline_root=self._pipeline_root,
                data_root=self._data_root,
                module_path=self._module_path,
                metadata_path=self._metadata_path,))

        expected_execution_count = 2

        metadata_config = (
            tfx.orchestration.metadata.sqlite_metadata_connection_config(
                self._metadata_path))
        with metadata.Metadata(metadata_config) as m:
            artifact_count = len(m.store.get_artifacts())
            execution_count = len(m.store.get_executions())
            self.assertGreaterEqual(artifact_count, execution_count)
            self.assertEqual(expected_execution_count, execution_count)

        self.assertPipelineExecution()



if __name__ == '__main__':

  tf.compat.v1.enable_v2_behavior()
  tf.test.main()

# _disabled pylint warning W0212: Access to a protected member till an alternate way is found
