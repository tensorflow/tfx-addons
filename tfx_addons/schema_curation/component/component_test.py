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
"""Tests for TFX Schema Curation Custom Component."""

import tensorflow as tf
from tfx.types import channel_utils, standard_artifacts

from tfx_addons.schema_curation.component import component


class SchemaCurationTest(tf.test.TestCase):
  def testConstruct(self):
    schema_curation = component.SchemaCuration(schema=channel_utils.as_channel(
        [standard_artifacts.Schema()]), )
    self.assertEqual(standard_artifacts.Schema.TYPE_NAME,
                     schema_curation.outputs['custom_schema'].type_name)


if __name__ == '__main__':
  tf.test.main()
