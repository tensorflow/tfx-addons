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
from typing import Any, Dict, List

from absl import logging
from kfp.pipeline_spec import pipeline_spec_pb2
from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.utils import proto_utils
from tfx_addons.message_exit_handler.message_providers import (
    LoggingMessageProvider,
    SlackMessageProvider,
)
from tfx_addons.message_exit_handler.spec import MessageExitHandlerSpec, MessagingType


class Executor(base_beam_executor.BaseBeamExecutor):
    """TFX Addons Message Exit Handler executor."""

    @staticmethod
    def Do(
        self,
        input_dict: Dict[str, List[types.Artifact]],
        output_dict: Dict[str, List[types.Artifact]],
        exec_properties: Dict[str, Any],
    ) -> None:

        self._log_startup(input_dict, output_dict, exec_properties)

        # parse the final status
        final_status = exec_properties.get(MessageExitHandlerSpec.FINAL_STATUS_KEY)
        pipeline_task_status = pipeline_spec_pb2.PipelineTaskFinalStatus()
        proto_utils.json_to_proto(
            exec_properties.get(final_status, pipeline_task_status)
        )
        logging.info(f"MessageExitHandler: {final_status}")
        status = json.loads(final_status)

        on_failure_only = exec_properties.get(
            MessageExitHandlerSpec.ON_FAILURE_ONLY_KEY
        )

        # leave the exit handler if pipeline succeeded and on_failure_only is True
        if on_failure_only and status["state"] == "SUCCEEDED":
            logging.info("MessageExitHandler: Skipping notification on success.")
            return

        message = self._set_message(status)

        message_type = exec_properties.get(MessageExitHandlerSpec.MESSAGE_TYPE_KEY)

        if message_type == MessagingType.SLACK.value:
            credentials = exec_properties[MessageExitHandlerSpec.SLACK_CREDENTIALS_KEY]
            provider = SlackMessageProvider(message, credentials)
        else:
            provider = LoggingMessageProvider(message)

        provider.send_message()
        logging.info(f"MessageExitHandler: {message}")
