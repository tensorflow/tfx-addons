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

import mock
import tensorflow as tf

mock.patch("tfx.orchestration.kubeflow.v2.decorators.exit_handler", lambda x: x).start()

from tfx_addons.message_exit_handler import component_fn
from tfx_addons.message_exit_handler import constants


class ComponentTest(tf.test.TestCase):
    @staticmethod
    def get_final_status(state: str = constants.SUCCESS_STATUS, error: str = "") -> str:
        # final status proto
        _status = {
            "state": state,
            "error": error,
            "pipelineJobResourceName": "projects/test-project/locations/us-central1/pipelineJobs/test-pipeline-job",
        }
        if error:
            _status.update({"error": {"message": error}})
        return _status

    def test_component_fn(self):

        final_status = self.get_final_status()

        with self.assertLogs(level="INFO") as logs:
            component_fn.SlackExitHandlerComponent(
                final_status=final_status, on_failure_only=True
            )

            self.assertLen(logs.output, 1)
            self.assertEqual(
                "INFO:absl:MessageExitHandler: Skipping notification on success.",
                logs.output[0],
            )


if __name__ == "__main__":
    tf.test.main()
