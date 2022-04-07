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
"""Message Providers supported by the Message Exit Handler component.

Currently supported:
* Logging
* Slack

"""

import enum
from typing import Dict, Text

from tfx_addons.message_exit_handler import constants


class MessagingType(enum.Enum):
  """Determines the type of message to send."""

  LOGGING = "logging"
  SLACK = "slack"


class BaseProvider:
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
