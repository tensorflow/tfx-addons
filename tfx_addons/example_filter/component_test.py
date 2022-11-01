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
"""Component test for the sampling component."""

import tensorflow as tf
from absl.testing import absltest
from tfx.types import artifact_utils, channel_utils, standard_artifacts
from . import component

class ComponentTest(absltest.TestCase):

    def testConstructWithOptions(self):
        examples = standard_artifacts.Examples()
        examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
        params = {
            "input_data": channel_utils.as_channel([examples]),
        "filter_function_str": 'filter_function',
        }

        print(component.FilterComponent(**params))


if __name__ == '__main__':
    #tf.test.main()

    component.FilterComponent(**params)