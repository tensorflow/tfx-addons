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
"""Message Providers supported by the Message Exit Handler component.

Currently supported:
* Logging
* Slack

"""

import enum
from typing import Dict, Optional, Text

from absl import logging
from slack import WebClient
from slack.errors import SlackApiError
from tfx.utils import import_utils
from tfx_addons.message_exit_handler import constants


class MessagingType(enum.Enum):
  """Determines the type of message to send."""

  LOGGING = "logging"
  SLACK = "slack"


class MessageProvider:
  """Message provider interface."""
  def __init__(self, status: Dict) -> None:
    self._status = status
    self._message = self.set_message(status)

  @staticmethod
  def set_message(status) -> str:
    """Set the message to be sent."""
    # parse the Vertex paths
    # structure: projects/{project}/locations/{location}/pipelineJobs/{pipeline_job}
    elements = status["pipelineJobResourceName"].split("/")
    project = elements[1]
    location = elements[3]
    job_id = elements[-1]

    # Generate message
    if status["state"] == constants.SUCCESS_STATUS:
      message = (
          ":tada: "
          f"Pipeline job *{job_id}* ({project}) completed successfully.\n")
    else:
      message = f":scream: Pipeline job *{job_id}* ({project}) failed."
      message += f"\n>{status['error']['message']}"

    message += f"\nhttps://console.cloud.google.com/vertex-ai/locations/{location}/pipelines/runs/{job_id}"
    return message

  def get_message(self) -> Text:
    """Get the message to be sent."""
    return self._message


class LoggingMessageProvider(MessageProvider):
  """Logging message provider."""
  def __init__(
      self,
      status: Dict,
      log_level: Optional[int] = logging.INFO,
  ) -> None:
    super().__init__(status=status)
    self._log_level = log_level

  def send_message(self) -> None:
    logging.log(self._log_level, f"MessageExitHandler: {self._message}")


class SlackMessageProvider(MessageProvider):
  """Slack message provider."""
  def __init__(self,
               status: Dict,
               credentials: str,
               decrypt_fn: Optional[str] = None) -> None:
    super().__init__(status=status)

    if not credentials:
      raise ValueError("Slack credentials not provided.")

    if decrypt_fn:
      module_path, fn_name = decrypt_fn.rsplit(".", 1)
      logging.info(
        f"MessageExitHandler: Importing {fn_name} from {module_path} "
        "to decrypt credentials.")
      fn = import_utils.import_func_from_module(module_path, fn_name)
      self._slack_channel_id = fn(credentials.slack_channel_id)
      self._slack_token = fn(credentials.slack_token)
    else:
      self._slack_channel_id = credentials.slack_channel_id
      self._slack_token = credentials.slack_token

    self._client = WebClient(token=self._slack_token)

  def send_message(self) -> None:
    try:
      response = self._client.chat_postMessage(channel=self._slack_channel_id,
                                               text=self._message)
      logging.info(f"MessageExitHandler: Slack response: {response}")
    except SlackApiError as e:
      logging.error(
          f"MessageExitHandler: Slack API error: {e.response['error']}")
