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
""" Message Exit Handler component """

import json

from absl import logging
from kfp.pipeline_spec import pipeline_spec_pb2
from tfx import v1 as tfx
from tfx.orchestration.kubeflow.v2.decorators import exit_handler
from tfx.utils import proto_utils

from tfx_addons.message_exit_handler import constants
from tfx_addons.message_exit_handler.message_providers.base_provider import \
    MessagingType
from tfx_addons.message_exit_handler.message_providers.logging_provider import \
    LoggingMessageProvider
from tfx_addons.message_exit_handler.message_providers.slack_provider import \
    SlackMessageProvider


@exit_handler
def MessageExitHandler(
    final_status: tfx.dsl.components.Parameter[str],
    on_failure_only: tfx.dsl.components.Parameter[bool] = False,
    message_type: tfx.dsl.components.Parameter[str] = MessagingType.LOGGING.
    value,
    slack_credentials: tfx.dsl.components.Parameter[str] = None,
    decrypt_fn: tfx.dsl.components.Parameter[str] = None,
):
  """
    Exit handler component for TFX pipelines originally developed by
    Digits Financial, Inc.
    The handler notifies the user of the final pipeline status via Slack.

    Args:
        final_status: The final status of the pipeline.
        slack_credentials: (Optional) The credentials to use for the
                           Slack API calls, json format.
        on_failure_only: (Optional) Whether to notify only on failure.
            False is the default.
        message_type: (Optional) The type of message to send.
            Logging is the default.
        decrypt_fn: (Optional) The function to use to decrypt the credentials,
        'tfx_addons.message_exit_handler.component_tests.fake_decryption_fn'

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
    # parse slack credentials
    if not slack_credentials:
      raise ValueError("Slack credentials not provided.")
    provider = SlackMessageProvider(status=status,
                                    credentials=slack_credentials,
                                    decrypt_fn=decrypt_fn)
  elif message_type == MessagingType.LOGGING.value:
    provider = LoggingMessageProvider(status=status)
  else:
    raise ValueError(
        f"MessageExitHandler: Unknown message type: {message_type}")

  provider.send_message()
  message = provider.get_message()
  logging.info(f"MessageExitHandler: {message}")
