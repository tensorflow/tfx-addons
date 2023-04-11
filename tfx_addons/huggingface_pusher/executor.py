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
"""HF Pusher TFX Component Executor. The HF Pusher Executor calls
the workflow handler runner.deploy_model_for_hf_hub().
"""

import ast
import time
from shutil import which
from typing import Any, Dict, List

from absl import logging
from tfx import types
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.types import artifact_utils, standard_component_specs
from tfx.utils import import_utils

from tfx_addons.huggingface_pusher import runner

_USERNAME_KEY = "username"
_ACCESS_TOKEN_KEY = "access_token"
_REPO_NAME_KEY = "repo_name"
_SPACE_CONFIG_KEY = "space_config"
_DECRYPT_FN_KEY = "decrypt_fn"


class Executor(tfx_pusher_executor.Executor):
  """Pushes a model and an app to HuggingFace Model and Space Hubs respectively"""
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
        - pushed_model: a TFX output channel containing a PushedModel arti
            fact. It contains information where the model is published at an
            d whether the model is pushed or not. furthermore, pushed model
            carries the following information.
            - pushed : integer value to denote if the model is pushed or not.
            This is set to 0 when the input model is not blessed, and it is
            set to 1 when the model is successfully pushed.
            - pushed_version : string value to indicate the current model ver
            sion. This is decided by time.time() Python built-in function.
            - repo_id : model repository ID where the model is pushed to. This
            follows the format of f"{username}/{repo_name}".
            - branch : branch name where the model is pushed to. The branch na
            me is automatically assigned to the same value of pushed_version.
            - commit_id : the id from the commit history (branch name could be
            sufficient to retreive a certain version of the model) of the mo
            del repository.
            - repo_url : model repository URL. It is something like f"https://
            huggingface.co/{repo_id}/{branch}"
            - space_url : space repository URL. It is something like f"https://
            huggingface.co/{repo_id}"f
        exec_properties: An optional dict of execution properties, including:
        - username: username of the HuggingFace user (can be an individual
            user or an organization)
        - access_token: access token value issued by HuggingFace for the s
            pecified username.
        - repo_name: the repository name to push the current version of the
            model to. The default value is same as the TFX pipeline name.
        - space_config: space_config carries additional values such as:
            - app_path : path where the application templates are in the cont
            ainer that runs the TFX pipeline. This is expressed either apps.
            gradio.img_classifier or apps/gradio.img_classifier.
            - repo_name : the repository name to push the application to. The
            default value is same as the TFX pipeline name
            - space_sdk : either gradio or streamlit. this will decide which a
            pplication framework to be used for the Space repository. The de
            fault value is gradio
            - placeholders : dictionary which placeholders to replace with mod
            el specific information. The keys represents describtions, and t
            he values represents the actual placeholders to replace in the f
            iles under the app_path. There are currently two predefined keys,
            and if placeholders is set to None, the default values will be used.
            - decrypt_fn: access token decryption function name including the m
            odule where it belongs to such as module_path.decrypt_fn.
        Raises:
            RuntimeError: if app_path is not set when space_config is provided.
            RuntimeError: if git-lfs is not installed.
        """
    self._log_startup(input_dict, output_dict, exec_properties)

    if self._is_git_lfs_installed() is not True:
      raise RuntimeError(
          "Git-LFS is not installed. "
          "Git-LFS installation guide can be found in "
          "https://huggingface.co/docs/hub/repositories-getting-started#requirements "
          "and https://git-lfs.github.com/.")

    decrypt_fn = exec_properties.get(_DECRYPT_FN_KEY, None)
    access_token = exec_properties.get(_ACCESS_TOKEN_KEY)

    if decrypt_fn:
      module_path, fn_name = decrypt_fn.rsplit(".", 1)
      logging.info(f"HFPusher: Importing {fn_name} from {module_path} "
                   "to decrypt credentials.")
      fn = import_utils.import_func_from_module(module_path, fn_name)
      access_token = fn(access_token)

    model_push = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.PUSHED_MODEL_KEY])

    # if the model is not blessed
    if not self.CheckBlessing(input_dict):
      self._MarkNotPushed(model_push)
      return

    model_path = self.GetModelPath(input_dict)
    model_version_name = f"v{int(time.time())}"

    space_config = exec_properties.get(_SPACE_CONFIG_KEY, None)
    if space_config is not None:
      space_config = ast.literal_eval(space_config)

    pushed_properties = runner.deploy_model_for_hf_hub(
        username=exec_properties.get(_USERNAME_KEY),
        access_token=access_token,
        repo_name=exec_properties.get(_REPO_NAME_KEY),
        space_config=space_config,
        model_path=model_path,
        model_version=model_version_name,
    )

    self._MarkPushed(model_push,
                     pushed_destination=pushed_properties["repo_url"])
    for key in pushed_properties:
      value = pushed_properties[key]

      if key != "repo_url":
        model_push.set_string_custom_property(key, value)

  def _is_git_lfs_installed(self):
    return which("git-lfs") is not None
