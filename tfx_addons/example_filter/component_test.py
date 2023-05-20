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
"""Component test for the filter component."""

import os

import tensorflow as tf
from absl.testing import absltest
from tfx.types import artifact_utils, standard_artifacts

from tfx_addons.example_filter.component import filter_component


class ComponentTest(absltest.TestCase):
  def testConstructWithOptions(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, "example_gen")
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    params = {
        "input_data": examples,
        "filter_function_str": 'filter_function',
        "output_file": 'output',
    }
    filter_component(**params)


if __name__ == '__main__':
    tf.test.main()
