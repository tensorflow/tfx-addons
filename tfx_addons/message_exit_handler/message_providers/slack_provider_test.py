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
"""Tests for Slack Provider functions."""

from unittest.mock import patch

import tensorflow as tf

from tfx_addons.message_exit_handler import constants
from tfx_addons.message_exit_handler.message_providers import slack_provider


class SlackProviderTest(tf.test.TestCase):
  @staticmethod
  def get_final_status(state: str = constants.SUCCESS_STATUS,
                       error: str = "") -> str:
    """Assemble final status for tests"""
    status = {
        "state":
        state,
        "error":
        error,
        "pipelineJobResourceName":
        ("projects/test-project/locations/"
         "us-central1/pipelineJobs/test-pipeline-job"),
    }
    if error:
      status.update({"error": {"message": error}})
    return status

  @patch(
      'tfx_addons.message_exit_handler.message_providers.slack_provider.WebClient'
  )
  def test_slack_message_provider(self, web_client_mock):
    final_status = self.get_final_status()
    credentials = slack_provider.SlackCredentials(
        slack_token="test-token", slack_channel_id="test-channel").json()

    message_provider = slack_provider.SlackMessageProvider(
        final_status, credentials)
    message_provider.send_message()
    web_client_mock.assert_called_once()
    web_client_mock.assert_called_with(token='test-token')

  @patch(
      'tfx_addons.message_exit_handler.message_providers.slack_provider.WebClient'
  )
  def test_slack_message_provider_with_decrypt_fn(self, mock_web_client):
    final_status = self.get_final_status()
    credentials = slack_provider.SlackCredentials(
        slack_token="test-token", slack_channel_id="test-channel").json()

    message_provider = slack_provider.SlackMessageProvider(
        final_status,
        credentials,
        decrypt_fn=
        'tfx_addons.message_exit_handler.component_test.fake_decryption_fn')
    message_provider.send_message()
    mock_web_client.assert_called_once()
    mock_web_client.assert_called_with(token='TEST-TOKEN')


if __name__ == "__main__":
  tf.test.main()
