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
""" Message provider interface for logging messages. """

from typing import Dict, Optional

from absl import logging

from tfx_addons.message_exit_handler.message_providers.base_provider import \
    BaseProvider


class LoggingMessageProvider(BaseProvider):
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
