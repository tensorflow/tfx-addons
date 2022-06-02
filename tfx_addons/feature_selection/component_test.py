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
"""Tests for tfx_addons.feature_selection.component"""

import json

import tensorflow as tf
from tfx_addons.feature_selection import component

import os
import tempfile
import urllib

import tensorflow as tf
tf.get_logger().propagate = False

from tfx import v1 as tfx
from tfx.components import CsvExampleGen
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

from tfx.types import standard_artifacts


# getting the dataset
_data_root = tempfile.mkdtemp(prefix='tfx-data')
DATA_PATH = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
             
_data_filepath = os.path.join(_data_root, "data.csv")
urllib.request.urlretrieve(DATA_PATH, _data_filepath)

context = InteractiveContext()

#create and run exampleGen component
example_gen = CsvExampleGen(input_base=_data_root )
context.run(example_gen)

#create and run statisticsGen component
statistics_gen = tfx.components.StatisticsGen(
    examples=example_gen.outputs['examples'])
context.run(statistics_gen)


feature_selection_comp = component.FeatureSelection(orig_examples = example_gen.outputs['examples'],
                                   module_file='example.modules.iris_module_file')
context.run(feature_selection_comp)


def test_feature_selection_artifact():
    assert isinstance(feature_selection_comp.outputs['feature_selection']._artifacts[0], component.FeatureSelectionArtifact)
    return


def test_output_example_artifact():
    assert isinstance(feature_selection_comp.outputs['updated_data']._artifacts[0], standard_artifacts.Examples)
    return

if __name__ == '__main__':
  tf.test.main()
