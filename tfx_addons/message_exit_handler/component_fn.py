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

import json
from typing import Optional

from absl import logging
from kfp.pipeline_spec import pipeline_spec_pb2
from tfx import v1 as tfx
from tfx.orchestration.kubeflow.v2.decorators import exit_handler
from tfx.utils import proto_utils

from tfx_addons.message_exit_handler import constants
from tfx_addons.message_exit_handler.message_providers import (
    LoggingMessageProvider,
    MessagingType,
    SlackMessageProvider,
)


@exit_handler
def SlackExitHandlerComponent(
    final_status: tfx.dsl.components.Parameter[str],
    on_failure_only: tfx.dsl.components.Parameter[bool] = False,
    message_type: Optional[str] = MessagingType.LOGGING.value,
    slack_credentials: Optional[str] = None,
):
  """
    Exit handler component for TFX pipelines originally developed by Digits Financial, Inc.
    The handler notifies the user of the final pipeline status via Slack.

    Args:
        final_status: The final status of the pipeline.
        slack_credentials: (Optional) The credentials to use for the Slack API calls.
        on_failure_only: Whether to notify only on failure
                        (default is 0, TFX < 1.6 doesn't support the boolean type).
        message_type: The type of message to send.

    """

  # parse the final status
  pipeline_task_status = pipeline_spec_pb2.PipelineTaskFinalStatus()
  proto_utils.json_to_proto(final_status, pipeline_task_status)
  logging.debug(f"MessageExitHandler: {final_status}")
  status = json.loads(final_status)

  # leave the exit handler if pipeline succeeded and on_failure_only is True
  if on_failure_only and status["state"] == constants.SUCCESS_STATUS:
    logging.info("MessageExitHandler: Skipping notification on success.")
    return

  # create the message provider
  if message_type == MessagingType.SLACK.value:
    provider = SlackMessageProvider(status=status,
                                    slack_credentials=slack_credentials)
  elif message_type == MessagingType.LOGGING.value:
    provider = LoggingMessageProvider(status=status)
  else:
    raise ValueError(
        f"MessageExitHandler: Unknown message type: {message_type}")

  provider.send_message()
  message = provider.get_message()
  logging.info(f"MessageExitHandler: {message}")
