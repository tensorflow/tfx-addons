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
""" Message provider interface for slack messages. """

from typing import Dict, Optional

from absl import logging
from pydantic import BaseModel
from slack import WebClient
from slack.errors import SlackApiError
from tfx.utils import import_utils

from tfx_addons.message_exit_handler.message_providers.base_provider import \
    BaseProvider


class SlackCredentials(BaseModel):
  """Pydantic class to de/serialize the slack credentials."""
  slack_token: str
  slack_channel_id: str


class SlackMessageProvider(BaseProvider):
  """Slack message provider."""
  def __init__(self,
               status: Dict,
               credentials: str,
               decrypt_fn: Optional[str] = None) -> None:
    super().__init__(status=status)

    if not credentials:
      raise ValueError("Slack credentials not provided.")

    credentials = SlackCredentials.parse_raw(credentials)
    self._slack_channel_id = credentials.slack_channel_id
    self._slack_token = credentials.slack_token

    if decrypt_fn:
      module_path, fn_name = decrypt_fn.rsplit(".", 1)
      logging.info(
          f"MessageExitHandler: Importing {fn_name} from {module_path} "
          "to decrypt credentials.")
      fn = import_utils.import_func_from_module(module_path, fn_name)
      self._slack_channel_id = fn(self._slack_channel_id)
      self._slack_token = fn(self._slack_token)

    self._client = WebClient(token=self._slack_token)

  def send_message(self) -> None:
    try:
      response = self._client.chat_postMessage(channel=self._slack_channel_id,
                                               text=self._message)
      logging.info(f"MessageExitHandler: Slack response: {response}")
    except SlackApiError as e:
      logging.error(
          f"MessageExitHandler: Slack API error: {e.response['error']}")
