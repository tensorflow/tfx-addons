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
"""Tests for TFX HuggingFace Pusher Custom Component."""

import tensorflow as tf
from tfx.types import channel_utils, standard_artifacts

from tfx_addons.huggingface_pusher.component import HFPusher


class HFPusherTest(tf.test.TestCase):
  def testConstruct(self):
    test_model = channel_utils.as_channel([standard_artifacts.Model()])
    hf_pusher = HFPusher(
        username="test_username",
        access_token="test_access_token",
        repo_name="test_repo_name",
        model=test_model,
        space_config={
            "repo_name": "test_repo_name",  # default: same as model repo_name
            "app_path": "app.gradio",  # or app/gradio
            "space_sdk": "gradio",  # or streamlit, default: gradio
            "placeholders": {
                # look for $MODEL_REPO_ID, $MODEL_REPO_URL, $MODEL_VERSION
                # tokens in files to replace with appropriate values at runtime
                "MODEL_REPO_ID": "$MODEL_REPO_ID",
                "MODEL_REPO_URL": "$MODEL_REPO_URL",
                "MODEL_VERSION": "$MODEL_VERSION",
            }
        })

    self.assertEqual(
        standard_artifacts.PushedModel.TYPE_NAME,
        hf_pusher.outputs["pushed_model"].type_name,
    )


if __name__ == "__main__":
  tf.test.main()
