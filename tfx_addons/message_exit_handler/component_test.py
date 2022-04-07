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
"""Tests for tfx_addons.message_exit_handler.component."""

import json
import unittest.mock as mock

import pytest
import tensorflow as tf
from tfx import v1 as tfx

from tfx_addons.message_exit_handler import constants
from tfx_addons.message_exit_handler.message_providers import base_provider
from tfx_addons.utils.test_utils import get_tfx_version


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

  @pytest.mark.skipif(get_tfx_version(tfx.__version__) < (1, 6, 0),
                      reason="not supported version")
  @mock.patch("tfx.orchestration.kubeflow.v2.decorators.exit_handler",
              lambda x: x)
  def test_component_fn(self):

    # import in function to pass TFX 1.4 tests
    from tfx_addons.message_exit_handler import \
        component  # pylint: disable=import-outside-toplevel
    final_status = self.get_final_status()

    with self.assertLogs(level="INFO") as logs:
      component.MessageExitHandler(final_status=final_status,
                                   on_failure_only=True)

      self.assertLen(logs.output, 1)
      self.assertEqual(
          "INFO:absl:MessageExitHandler: Skipping notification on success.",
          logs.output[0],
      )

  @pytest.mark.skipif(get_tfx_version(tfx.__version__) < (1, 6, 0),
                      reason="not supported version")
  @mock.patch(
      "tfx_addons.message_exit_handler.message_providers.slack_provider.WebClient"  # pylint: disable=line-too-long
  )
  @mock.patch("tfx.orchestration.kubeflow.v2.decorators.exit_handler",
              lambda x: x)
  def test_component_slack(self, mock_web_client):

    # import in function to pass TFX 1.4 tests
    from tfx_addons.message_exit_handler import \
        component  # pylint: disable=import-outside-toplevel
    final_status = self.get_final_status()
    creds = json.dumps({"slack_token": "token", "slack_channel_id": "channel"})

    with self.assertLogs(level="INFO"):
      component.MessageExitHandler(
          final_status=final_status,
          message_type=base_provider.MessagingType.SLACK.value,
          slack_credentials=creds,
      )

      mock_web_client.assert_called_once()
      mock_web_client.assert_called_with(token="token")

  @pytest.mark.skipif(get_tfx_version(tfx.__version__) < (1, 6, 0),
                      reason="not supported version")
  @mock.patch(
      "tfx_addons.message_exit_handler.message_providers.slack_provider.WebClient"  # pylint: disable=line-too-long
  )
  @mock.patch("tfx.orchestration.kubeflow.v2.decorators.exit_handler",
              lambda x: x)
  def test_component_slack_decrypt(self, mock_web_client):

    # import in function to pass TFX 1.4 tests
    from tfx_addons.message_exit_handler import \
        component  # pylint: disable=import-outside-toplevel
    final_status = self.get_final_status()
    creds = json.dumps({"slack_token": "token", "slack_channel_id": "channel"})

    with self.assertLogs(level="INFO"):
      component.MessageExitHandler(
          final_status=final_status,
          message_type=base_provider.MessagingType.SLACK.value,
          slack_credentials=creds,
          decrypt_fn=
          "tfx_addons.message_exit_handler.component_test.fake_decryption_fn",
      )

      mock_web_client.assert_called_once()
      mock_web_client.assert_called_with(token="TOKEN")


if __name__ == "__main__":
  tf.test.main()
