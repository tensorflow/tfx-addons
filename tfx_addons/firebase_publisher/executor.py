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
"""Firebase Publisher TFX Component Executor.

The Firebase Publisher Executor calls the workflow handler
runner.deploy_model_for_firebase_ml().
"""

import time
from typing import Any, Dict, List

from tfx import types
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.types import artifact_utils, standard_component_specs

from tfx_addons.firebase_publisher import runner

_APP_NAME_KEY = "app_name"
_DISPLAY_NAME_KEY = "display_name"
_STORAGE_BUCKET_KEY = "storage_bucket"
_TAGS_KEY = "tags"
_OPTIONS_KEY = "options"
_CREDENTIAL_PATH_KEY = "credential_path"


class Executor(tfx_pusher_executor.Executor):
  """Pushes a model to Firebase ML."""
  def Do(
      self,
      input_dict: Dict[str, List[types.Artifact]],
      output_dict: Dict[str, List[types.Artifact]],
      exec_properties: Dict[str, Any],
  ):
    """Overrides the tfx_pusher_executor to leverage some of utility methods

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: a TFX input channel containing a Model artifact.
        - model_blessing: a TFX input channel containing a ModelBlessing
          artifact.
      output_dict: Output dict from key to a list of artifacts, including:
        - pushed_model: a TFX output channel containing a PushedModel artifact.
          It contains information where the model is published at and whether
          the model is pushed or not.
      exec_properties: An optional dict of execution properties, including:
        - display_name: name to identify a hosted model in Firebase ML.
            this should be a unique value because it will be used to search
            a existing model to update.
        - storage_bucket: GCS bucket where the hosted model will be stored.
        - app_name: the name of Firebase app to determine the scope.
        - tags: tags to be attached to the hosted ML model.
        - credential_path: location of GCS or local file system where the
          Service Account(SA) Key file is.
        - options: additional configurations to be passed to initialize Firebase
          app.

    Raises:
      RuntimeError: when the size of model exceeds 40mb.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    model_push = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.PUSHED_MODEL_KEY])
    if not self.CheckBlessing(input_dict):
      self._MarkNotPushed(model_push)
      return
    model_path = self.GetModelPath(input_dict)
    model_version_name = f"v{int(time.time())}"

    pushed_model_path = runner.deploy_model_for_firebase_ml(
        app_name=exec_properties.get(_APP_NAME_KEY, '[DEFAULT]'),
        display_name=exec_properties.get(_DISPLAY_NAME_KEY),
        storage_bucket=exec_properties.get(_STORAGE_BUCKET_KEY),
        credential_path=exec_properties.get(_CREDENTIAL_PATH_KEY, None),
        tags=exec_properties.get(_TAGS_KEY, []),
        options=exec_properties.get(_OPTIONS_KEY, {}),
        model_path=model_path,
        model_version=model_version_name,
    )

    self._MarkPushed(model_push, pushed_destination=pushed_model_path)
