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
"""Message Exit Handler component definition."""

from typing import Optional, Union

from tfx.dsl.components.base import base_beam_component, executor_spec

from tfx_addons.message_exit_handler.executor import Executor
from tfx_addons.message_exit_handler.proto import slack_pb2
from tfx_addons.message_exit_handler.spec import MessageExitHandlerSpec, MessagingType


class MessageExitHandler(base_beam_component.BaseBeamComponent):
  """Exit handler component for TFX pipelines originally developed by Digits Financial, Inc.
    The handler notifies the user of the final pipeline status via Slack.

    Args:
        final_status: The final status of the pipeline.
        slack_credentials: (Optional) The credentials to use for the Slack API calls.
        on_failure_only: Whether to notify only on failure
                        (default is 0, TFX < 1.6 doesn't support the boolean type).
        message_type: The type of message to send.

    Returns:
        None

    ## Example Use

    # TODO: Add example setup
    ...

    """

  SPEC_CLASS = MessageExitHandlerSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(Executor)

  def __init__(
      self,
      final_status: str,
      message_type: MessagingType = MessagingType.LOGGING,
      slack_credentials: Optional[slack_pb2.SlackSpec] = None,
      failure_only: Optional[Union[int, bool]] = 0,
  ):
    """Construct a MessageExitHandler component.
        Args:
            final_status: The final status of the pipeline.
            slack_credentials: (Optional) The credentials to use for the Slack API calls.
            on_failure_only: Whether to notify only on failure
                            (default is 0, TFX < 1.6 doesn't support the boolean type).
            message_type: The type of message to send.

        """

    _failure_only = int(failure_only) if failure_only else 0
    spec = MessageExitHandlerSpec(
        final_status=final_status,
        message_type=message_type,
        slack_credentials=slack_credentials or slack_pb2.SlackSpec(),
        failure_only=_failure_only,
    )

    super().__init__(spec=spec)
