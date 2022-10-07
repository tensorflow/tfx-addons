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
"""Tests for tfx_addons.huggingface_pusher.executor."""

from unittest import mock
from unittest.mock import ANY, Mock

import tensorflow as tf
from tfx.types import standard_artifacts

from tfx_addons.huggingface_pusher import executor


class RunnerTest(tf.test.TestCase):
  @mock.patch('tfx_addons.huggingface_pusher.executor.which')
  @mock.patch('tfx_addons.huggingface_pusher.runner.deploy_model_for_hf_hub')
  def testCheckGitLFSInstalled(self, mock_runner_function, mock_which):
    executor.Executor.CheckBlessing = Mock()
    executor.Executor.GetModelPath = Mock()
    executor.Executor._MarkPushed = Mock()  # pylint: disable=protected-access

    executor.Executor.CheckBlessing.return_value = True
    executor.Executor.GetModelPath.return_value = "test_model_path"

    mock_runner_function.return_value = {
        "test_key": "test_value",
        "repo_url": "test_repo_url"
    }
    mock_which.return_value = "git-lfs"

    exe = executor.Executor()

    try:
      exe.Do({}, {"pushed_model": [standard_artifacts.PushedModel()]}, {})
    except RuntimeError:
      assert False

  @mock.patch('tfx_addons.huggingface_pusher.executor.which')
  @mock.patch('tfx_addons.huggingface_pusher.runner.deploy_model_for_hf_hub')
  def testCheckGitLFSNotInstalled(self, mock_runner_function, mock_which):
    executor.Executor.CheckBlessing = Mock()
    executor.Executor.GetModelPath = Mock()
    executor.Executor._MarkPushed = Mock()  # pylint: disable=protected-access

    executor.Executor.CheckBlessing.return_value = True
    executor.Executor.GetModelPath.return_value = "test_model_path"

    mock_runner_function.return_value = {
        "test_key": "test_value",
        "repo_url": "test_repo_url"
    }
    mock_which.return_value = None

    exe = executor.Executor()

    try:
      exe.Do({}, {"pushed_model": [standard_artifacts.PushedModel()]}, {})
      assert False
    except RuntimeError:
      assert True

  @mock.patch('tfx_addons.huggingface_pusher.executor.which')
  @mock.patch('tfx_addons.huggingface_pusher.runner.deploy_model_for_hf_hub')
  def testCheckDecyprtFn(self, mock_runner_function, mock_which):
    executor.Executor.CheckBlessing = Mock()
    executor.Executor.GetModelPath = Mock()
    executor.Executor._MarkPushed = Mock()  # pylint: disable=protected-access

    executor.Executor.CheckBlessing.return_value = True
    executor.Executor.GetModelPath.return_value = "test_model_path"

    mock_runner_function.return_value = {
        "test_key": "test_value",
        "repo_url": "test_repo_url"
    }
    mock_which.return_value = "git-lfs"

    exe = executor.Executor()

    exe.Do({}, {"pushed_model": [standard_artifacts.PushedModel()]}, {
        "access_token":
        "access_token",
        "decrypt_fn":
        "tfx_addons.huggingface_pusher.executor_test.fake_decryption_fn"
    })
    mock_runner_function.assert_called_with(username=ANY,
                                            repo_name=ANY,
                                            space_config=ANY,
                                            model_path=ANY,
                                            model_version=ANY,
                                            access_token="ACCESS_TOKEN")


def fake_decryption_fn(encrypted_message):
  return encrypted_message.upper()
