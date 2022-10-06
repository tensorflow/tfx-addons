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
"""Tests for tfx_addons.huggingface_pusher.runner."""

from unittest import mock
from unittest.mock import Mock

import tensorflow as tf

from tfx_addons.huggingface_pusher import runner


class RunnerTest(tf.test.TestCase):
  def testCheckWorkflowWithoutSpaceConfig(self):
    runner._create_remote_repo = Mock()  # pylint: disable=protected-access
    runner._clone_and_checkout = Mock()  # pylint: disable=protected-access
    runner._replace_files = Mock()  # pylint: disable=protected-access
    runner._push_to_remote_repo = Mock()  # pylint: disable=protected-access
    runner._replace_placeholders = Mock()  # pylint: disable=protected-access

    runner.deploy_model_for_hf_hub(
        username="test_username",
        access_token="test_access_token",
        repo_name="test_repo_name",
        model_path="test_model_path",
        model_version="test_model_version",
    )

    runner._create_remote_repo.assert_called_once()  # pylint: disable=protected-access
    runner._clone_and_checkout.assert_called_once()  # pylint: disable=protected-access
    runner._replace_files.assert_called_once()  # pylint: disable=protected-access
    runner._push_to_remote_repo.assert_called_once()  # pylint: disable=protected-access
    runner._replace_placeholders.assert_not_called()  # pylint: disable=protected-access

  def testCheckWorkflowWithSpaceConfigButWithoutAppPath(self):
    runner._create_remote_repo = Mock()  # pylint: disable=protected-access
    runner._clone_and_checkout = Mock()  # pylint: disable=protected-access
    runner._replace_files = Mock()  # pylint: disable=protected-access
    runner._push_to_remote_repo = Mock()  # pylint: disable=protected-access
    runner._replace_placeholders = Mock()  # pylint: disable=protected-access

    try:
      runner.deploy_model_for_hf_hub(username="test_username",
                                     access_token="test_access_token",
                                     repo_name="test_repo_name",
                                     model_path="test_model_path",
                                     model_version="test_model_version",
                                     space_config={})
    except RuntimeError:
      assert True

    runner._create_remote_repo.assert_called_once()  # pylint: disable=protected-access
    runner._clone_and_checkout.assert_called_once()  # pylint: disable=protected-access
    runner._replace_files.assert_called_once()  # pylint: disable=protected-access
    runner._push_to_remote_repo.assert_called_once()  # pylint: disable=protected-access
    runner._replace_placeholders.assert_not_called()  # pylint: disable=protected-access

  @mock.patch('tfx_addons.huggingface_pusher.runner.io_utils.copy_dir')
  def testCheckWorkflowWithSpaceConfigButWithAppPath(self, mock_copy_dir):
    runner._create_remote_repo = Mock()  # pylint: disable=protected-access
    runner._clone_and_checkout = Mock()  # pylint: disable=protected-access
    runner._replace_files = Mock()  # pylint: disable=protected-access
    runner._push_to_remote_repo = Mock()  # pylint: disable=protected-access
    runner._replace_placeholders = Mock()  # pylint: disable=protected-access

    try:
      runner.deploy_model_for_hf_hub(
          username="test_username",
          access_token="test_access_token",
          repo_name="test_repo_name",
          model_path="test_model_path",
          model_version="test_model_version",
          space_config={"app_path": "test_app_path"})
    except RuntimeError:
      assert False

    self.assertEqual(runner._create_remote_repo.call_count, 2)  # pylint: disable=protected-access
    self.assertEqual(runner._clone_and_checkout.call_count, 2)  # pylint: disable=protected-access
    self.assertEqual(runner._replace_files.call_count, 2)  # pylint: disable=protected-access
    self.assertEqual(runner._push_to_remote_repo.call_count, 2)  # pylint: disable=protected-access
    runner._replace_placeholders.assert_called_once()  # pylint: disable=protected-access
    mock_copy_dir.assert_called_once()


if __name__ == "__main__":
  tf.test.main()
