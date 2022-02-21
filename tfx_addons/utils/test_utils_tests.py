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
    TFX_VERSION = "1.4.0"
    self.assertEqual(
      test_utils.get_tfx_version(TFX_VERSION),
      [1, 4, 0])

  def test_disable_test_within_range(self):
    fn = test_utils.disable_test(
        test_fn, min_version="1.0.0", max_version="2.0.0")
    self.assertEqual(fn(), MESSAGE_FN_CALLED)
    self.assertEqual(fn.__name__, "test_fn")

  def test_disable_test_outside_range(self):
    with self.assertLogs(level="INFO") as logs:
      fn = test_utils.disable_test(
        test_fn, min_version="0.22.0", max_version="1.1.9")
      fn()
      self.assertEqual(len(logs.output), 1)
      self.assertEqual(
          EXPECTED_WARNING_MESSAGE, logs.output[0])
      self.assertEqual(fn.__name__, "_decorator")

if __name__ == "__main__":
  unittest.main()
