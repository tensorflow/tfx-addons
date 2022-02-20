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
"""Tests for tfx_addons.message_exit_handler.component."""

import json
import mock
import tensorflow as tf

mock.patch("tfx.orchestration.kubeflow.v2.decorators.exit_handler",
           lambda x: x).start()

from tfx_addons.message_exit_handler import component, constants
from tfx_addons.message_exit_handler import message_providers
from tfx_addons.message_exit_handler.proto import slack_pb2


def fake_decryption_fn(encrypted_message):
  return encrypted_message.upper()

class ComponentTest(tf.test.TestCase):
  @staticmethod
  def get_final_status(state: str = constants.SUCCESS_STATUS,
                       error: str = "") -> str:
    """Assemble final status for tests"""
    # final status proto
    status = {
        "state":
        state,
        "error":
        error,
        "pipelineJobResourceName":
        ("projects/test-project/locations/us-central1/"
         "pipelineJobs/test-pipeline-job"),
    }
    if error:
      status.update({"error": {"message": error}})
    return json.dumps(status)

  def test_component_fn(self):

    final_status = self.get_final_status()

    with self.assertLogs(level="INFO") as logs:
      component.MessageExitHandler(final_status=final_status,
                                   on_failure_only=True)

      self.assertLen(logs.output, 1)
      self.assertEqual(
          "INFO:absl:MessageExitHandler: Skipping notification on success.",
          logs.output[0],
      )

  @mock.patch('tfx_addons.message_exit_handler.message_providers.WebClient')
  def test_component_slack(self, mock_web_client):

    final_status = self.get_final_status()

    with self.assertLogs(level="INFO") as logs:
      component.MessageExitHandler(final_status=final_status,
                                   message_type=message_providers.MessagingType.SLACK.value,
                                   slack_credentials=slack_pb2.SlackSpec(
                                     slack_token="token",
                                     slack_channel_id="channel",)
      )

      mock_web_client.assert_called_once()
      mock_web_client.assert_called_with(token='token')


  @mock.patch('tfx_addons.message_exit_handler.message_providers.WebClient')
  def test_component_slack_decrypt(self, mock_web_client):

    final_status = self.get_final_status()

    with self.assertLogs(level="INFO") as logs:
      component.MessageExitHandler(final_status=final_status,
                                   message_type=message_providers.MessagingType.SLACK.value,
                                   slack_credentials=slack_pb2.SlackSpec(
                                     slack_token="token",
                                     slack_channel_id="channel",),
                                   decrypt_fn='tfx_addons.message_exit_handler.component_tests.fake_decryption_fn'
      )

      mock_web_client.assert_called_once()
      mock_web_client.assert_called_with(token='TOKEN')


if __name__ == "__main__":
  tf.test.main()
