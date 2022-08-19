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
import tempfile
# import urllib


# from tensorflow import test
# from tfx.components import CsvExampleGen
# from tfx.orchestration.experimental.interactive.interactive_context import \
    # InteractiveContext
# from tfx.types import standard_artifacts
# from tfx.dsl import Pipeline
# from tfx.orchestration import metadata
from typing import Text, Optional, List
import tfx
import tensorflow as tf

from tfx_addons.feature_selection import component


import ml_metadata as mlmd
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2


# getting the dataset
_data_root = tempfile.mkdtemp(prefix='tfx-data')
# DATA_PATH = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
# DATA_PATH = os.path.join("test", "iris.csv")


def _create_pipeline(
    pipeline_name: Text, 
    pipeline_root: Text, 
    data_root: Text, 
    module_file: Text,
    metadata_connection_config: Optional[
    metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None
) -> tfx.v1.dsl.Pipeline:

    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # creating and executing the FeatureSelection artifact
    feature_selection = component.FeatureSelection(
        orig_examples=example_gen.outputs['examples'],
        module_file=module_file)

    components = [
        example_gen,
        feature_selection
    ]

    return tfx.v1.dsl.Pipeline(
        pipeline_name=pipeline_name, 
        pipeline_root=pipeline_root, 
        components=components, 
        metadata_connection_config=metadata_connection_config,
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
        self._module_file = os.path.join(self._feature_selection_root, 'example', 'modules', 'iris_module_file.py')
        self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                        self._pipeline_name)
        
        self.connection_config = metadata_store_pb2.ConnectionConfig()

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

        self.connection_config.fake_database.SetInParent() # Sets an empty fake database proto.


        tfx.v1.orchestration.LocalDagRunner().run(
            _create_pipeline(
                pipeline_name=self._pipeline_name,
                pipeline_root=self._pipeline_root,
                data_root=self._data_root,
                module_file=self._module_file,
                metadata_connection_config=self.connection_config))

        expected_execution_count = 6

        # store = metadata_store.MetadataStore(self.connection_config)

        # artifact_count = len(store.get_artifacts())
        # execution_count = len(store.get_executions())
        # self.assertGreaterEqual(artifact_count, execution_count)
        # self.assertEqual(expected_execution_count, execution_count)

        self.assertPipelineExecution()



if __name__ == '__main__':

#   tf.compat.v1.enable_v2_behavior()
  tf.test.main()

# _disabled pylint warning W0212: Access to a protected member till an alternate way is found
