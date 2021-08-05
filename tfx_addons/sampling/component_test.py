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
from tfx.utils import json_utils

from tfx_addons.sampling import component, spec


class ComponentTest(absltest.TestCase):
  def testConstruct(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    params = {
        spec.SAMPLER_INPUT_KEY: channel_utils.as_channel([examples]),
        spec.SAMPLER_SPLIT_KEY: ['train'],
        spec.SAMPLER_LABEL_KEY: 'label'
    }

    under = component.Sampler(**params)

    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     under.outputs[spec.SAMPLER_OUTPUT_KEY].type_name)
    self.assertEqual(under.spec.exec_properties[spec.SAMPLER_SPLIT_KEY],
                     json_utils.dumps(['train']))
    self.assertEqual(under.spec.exec_properties[spec.SAMPLER_LABEL_KEY],
                     'label')

  def testConstructWithOptions(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    params = {
        spec.SAMPLER_INPUT_KEY: channel_utils.as_channel([examples]),
        spec.SAMPLER_LABEL_KEY: 'test_label',
        spec.SAMPLER_NAME_KEY: 'test_name',
        spec.SAMPLER_SPLIT_KEY: ['train', 'eval'],
        spec.SAMPLER_COPY_KEY: False,
        spec.SAMPLER_SHARDS_KEY: 10,
        spec.SAMPLER_CLASSES_KEY: ['label']
    }

    under = component.Sampler(**params)

    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     under.outputs[spec.SAMPLER_OUTPUT_KEY].type_name)
    self.assertEqual(under.spec.exec_properties[spec.SAMPLER_LABEL_KEY],
                     'test_label')
    self.assertEqual(under.spec.exec_properties[spec.SAMPLER_NAME_KEY],
                     'test_name')
    self.assertEqual(under.spec.exec_properties[spec.SAMPLER_SPLIT_KEY],
                     json_utils.dumps(['train', 'eval']))
    self.assertEqual(under.spec.exec_properties[spec.SAMPLER_COPY_KEY], False)
    self.assertEqual(under.spec.exec_properties[spec.SAMPLER_SHARDS_KEY], 10)
    self.assertEqual(under.spec.exec_properties[spec.SAMPLER_CLASSES_KEY],
                     json_utils.dumps(['label']))


if __name__ == '__main__':
  tf.test.main()
