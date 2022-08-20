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

import csv
import importlib
import os
from typing import List, Optional, Text

import tensorflow as tf
import tfx
from tfx.orchestration import metadata

from tfx_addons.feature_selection import component


def _get_selected_features(module_file, data_path):
  """Get the correct selected features for testing"""

  data = []

  # importing required configurations
  modules = importlib.import_module(module_file)
  mod_names = ["SELECTOR_PARAMS", "TARGET_FEATURE", "SelectorFunc"]
  selector_params, target_feature, selector_func = [
      getattr(modules, i) for i in mod_names
  ]

  # getting the data
  with open(data_path, 'r') as file:
    my_reader = csv.reader(file, delimiter=',')
    for row in my_reader:
      data.append(row)

  # splitting X (input) and Y (output) from CSV data
  target_idx = data[0].index(target_feature)
  target_data = [i.pop(target_idx) for i in data]

  # runnign the selector function for feature selection
  selector = selector_func(**selector_params)
  selector.fit_transform(data[1:], target_data[1:])

  # getting selected feature names
  selected_indices = selector.get_support(indices=True)
  final_features = set(data[0][idx] for idx in selected_indices)

  return final_features


def _create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_root: Text,
    module_path: Text,
    metadata_path: Text,
    beam_pipeline_args: Optional[List[Text]] = None) -> tfx.v1.dsl.Pipeline:
  """Creating sample pipeline with two components: CsvExampleGen and
  FeatureSelection"""

  # specifying the pipeline components
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)
  feature_selection = component.FeatureSelection(
      orig_examples=example_gen.outputs['examples'], module_path=module_path)

  components = [example_gen, feature_selection]

  return tfx.v1.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
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
    self._data_path = os.path.join(self._data_root, 'iris.csv')
    self._module_path = os.path.join(self._feature_selection_root, 'example',
                                     'modules', 'iris_module_file.py')
    self._module_file = "tfx_addons.feature_selection.example.modules.\
iris_module_file"

    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  def assertExecutedOnce(self, component: Text) -> None:  # pylint: disable=W0621
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
        _create_pipeline(pipeline_name=self._pipeline_name,
                         pipeline_root=self._pipeline_root,
                         data_root=self._data_root,
                         module_path=self._module_path,
                         metadata_path=self._metadata_path))

    expected_execution_count = 2  # one each for CsvExampleGen and Feature Selection
    true_selected_features = _get_selected_features(self._module_file,
                                                    self._data_path)

    metadata_config = (
        tfx.orchestration.metadata.sqlite_metadata_connection_config(
            self._metadata_path))
    with metadata.Metadata(metadata_config) as m:
      execution_count = len(m.store.get_executions())
      selected_features_struct = list(
          m.store.get_artifacts_by_type(
              "Feature Selection")[0].properties["selected_features"].
          struct_value.fields.values.__self__["__value__"].list_value.values)
      component_selected_features = set(
          feature.string_value for feature in selected_features_struct)

      # TEST: execution count
      self.assertEqual(expected_execution_count, execution_count)

      # TEST: number of artifacts with TYPE_NAME `Feature Selection`
      self.assertEqual(1,
                       len(m.store.get_artifacts_by_type("Feature Selection")))

      # TEST: number of artifacts with TYPE_NAME `Examples`
      # (one each from CsvExampleGen and FeatureSelection)
      self.assertEqual(2, len(m.store.get_artifacts_by_type("Examples")))

      # TEST: if the features selected by component are correct
      self.assertEqual(component_selected_features, true_selected_features)

    self.assertPipelineExecution()


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()

# _disabled pylint warning `W0621: Redefining name 'component' from outer scope` till an alternate way is found
