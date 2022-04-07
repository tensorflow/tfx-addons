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
"""Tests for TFX Addons test util functions."""

import unittest

from tfx_addons.utils import test_utils

MESSAGE_FN_CALLED = "test_fn called"
EXPECTED_WARNING_MESSAGE = (
    "WARNING:absl:test_fn has been disabled due to incompatible TFX version.")


def test_fn():
  return MESSAGE_FN_CALLED


class TestUtilTest(unittest.TestCase):
  def test_get_tfx_version(self):
    tfx_version = "1.4.0"
    self.assertEqual(test_utils.get_tfx_version(tfx_version), (1, 4, 0))


if __name__ == "__main__":
  unittest.main()
