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
"""Tests for Message Provider functions."""

import tensorflow as tf
from tfx_addons.message_exit_handler import constants, message_providers


SUCCESS_MESSAGE = """:tada: Pipeline job *test-pipeline-job* (test-project) completed successfully.

https://console.cloud.google.com/vertex-ai/locations/us-central1/pipelines/runs/test-pipeline-job"""

FAILURE_MESSAGE = """:scream: Pipeline job *test-pipeline-job* (test-project) failed.
>test error
https://console.cloud.google.com/vertex-ai/locations/us-central1/pipelines/runs/test-pipeline-job"""


class MessageProviderTest(tf.test.TestCase):
  @staticmethod
  def get_final_status(state: str = constants.SUCCESS_STATUS,
                       error: str = "") -> str:
    # final status proto
    _status = {
        "state":
        state,
        "error":
        error,
        "pipelineJobResourceName":
        "projects/test-project/locations/us-central1/pipelineJobs/test-pipeline-job",
    }
    if error:
      _status.update({"error": {"message": error}})
    return _status

  # def setUp(self):
  #     self.handler_fn = component_fn.SlackExitHandlerComponent

  def test_message_provider_success(self):

    final_status = self.get_final_status()
    test_provider = message_providers.MessageProvider(final_status)

    self.assertEqual(SUCCESS_MESSAGE, test_provider.get_message())

  def test_message_provider_failure(self):

    final_status = self.get_final_status(state=constants.FAILURE_STATUS,
                                         error="test error")
    test_provider = message_providers.MessageProvider(final_status)

    self.assertEqual(FAILURE_MESSAGE, test_provider.get_message())

  def test_logging_message_provider(self):

    final_status = self.get_final_status()
    with self.assertLogs(level="INFO") as logs:
      message_provider = message_providers.LoggingMessageProvider(final_status)
      message_provider.send_message()

      self.assertLen(logs.output, 1)
      self.assertEqual(
          "INFO:absl:MessageExitHandler: " + SUCCESS_MESSAGE,
          logs.output[0],
      )


if __name__ == "__main__":
  tf.test.main()
