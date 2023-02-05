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
"""HuggingFace(HF) Pusher TFX Component.
The HFPusher is used to push model and prototype application to HuggingFace Hub.
"""
from typing import Any, Dict, Optional, Text

from tfx import types
from tfx.dsl.components.base import base_component, executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter

from tfx_addons.huggingface_pusher import executor

MODEL_KEY = "model"
PUSHED_MODEL_KEY = "pushed_model"
MODEL_BLESSING_KEY = "model_blessing"


class HFPusherSpec(types.ComponentSpec):
  """ComponentSpec for TFX HFPusher Component."""

  PARAMETERS = {
      "username": ExecutionParameter(type=str),
      "access_token": ExecutionParameter(type=str),
      "repo_name": ExecutionParameter(type=str),
      "space_config": ExecutionParameter(type=Dict[Text, Any], optional=True),
      "decrypt_fn": ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      MODEL_KEY:
      ChannelParameter(type=standard_artifacts.Model, optional=True),
      MODEL_BLESSING_KEY:
      ChannelParameter(type=standard_artifacts.ModelBlessing, optional=True),
  }
  OUTPUTS = {
      PUSHED_MODEL_KEY: ChannelParameter(type=standard_artifacts.PushedModel),
  }


class HFPusher(base_component.BaseComponent):
  """Component for pushing model and application to HuggingFace Hub.

    The `HFPusher` is a [TFX Component](https://www.tensorflow.org/tfx
    /guide/understanding_tfx_pipelines#component), and its primary pur
    pose is to push a model from an upstream component such as [`Train
    er`](https://www.tensorflow.org/tfx/guide/trainer) to HuggingFace
    Model Hub. It also provides a secondary feature that pushes an app
    lication to HuggingFace Space Hub.
    """

  SPEC_CLASS = HFPusherSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      username: str,
      access_token: str,
      repo_name: str,
      space_config: Optional[Dict[Text, Any]] = None,
      decrypt_fn: Optional[str] = None,
      model: Optional[types.Channel] = None,
      model_blessing: Optional[types.Channel] = None,
  ):
    """The HFPusher TFX component.

        HFPusher pushes a trained or blessed model to HuggingFace Model Hub.
        This is designed to work as a downstream component of Trainer and o
        ptionally Evaluator(optional) components. Trainer gives trained mod
        el, and Evaluator gives information whether the trained model is bl
        essed or not after evaluation of the model. HFPusher component only
        publishes a model when it is blessed. If Evaluator is not specified,
        the input model will always be pushed.

        Args:
        username: the ID of HuggingFace Hub
        access_token: the access token obtained from HuggingFace Hub for the
            given username. Refer to [this document](https://huggingface.co/
            docs/hub/security-tokens) to know how to obtain one.
        repo_name: the name of Model Hub repository where the model will be
            pushed. This should be unique name under the username within th
            e Model Hub. repository is identified as {username}/{repo_name}.
        space_config: optional configurations set when to push an application
            to HuggingFace Space Hub. This is a dictionary, and the following
            information could be set.
            app_path: the path where the application related files are stored.
                this should follow the form either of app.gradio.segmentation
                or app/gradio/segmentation. This is a required parameter when
                space_config is set. This could be a local or GCS paths.
            space_sdk: Space Hub supports gradio, streamit, and static types
                of application. The default is set to gradio.
            placeholders: placeholders to replace in every files under the a
                pp_path. This is used to replace special string with the mod
                el related values. If this is not set, the default placehold
                ers will be used as follows.
                ```
                placeholders = {
                    "MODEL_REPO_ID" : "$MODEL_REPO_ID",
                    "MODEL_REPO_URL": "$MODEL_REPO_URL",
                    "MODEL_VERSION" : "$MODEL_VERSION",
                }
                ```
                In this case, "$MODEL_REPO_ID", "$MODEL_REPO_URL", "$MODEL_VE
                RSION" strings will be replaced with appropriate values at ru
                ntime. If placeholders are set, custom strings will be used.
            repo_name: the name of Space Hub repository where the application
                will be pushed. This should be unique name under the username
                within the Space Hub. repository is identified as {username}/
                {repo_name}. If this is not set, the same name to the Model H
                ub repository will be used.
        decrypt_fn: access token decryption function name including the module
            where it belongs to such as module_path.decrypt_fn.
        model: a TFX input channel containing a Model artifact. this is usually
            comes from the standard [`Trainer`]
            (https://www.tensorflow.org/tfx/guide/trainer) component.
        model_blessing: a TFX input channel containing a ModelBlessing artifact.
            this is usually comes from the standard [`Evaluator`]
            (https://www.tensorflow.org/tfx/guide/evaluator) component.
        Returns:
        a TFX output channel containing a PushedModel artifact. It contains
        information where the model is published at and whether the model is
        pushed or not.

        Raises:
            RuntimeError: if app_path is not set when space_config is provided.
            RuntimeError: if git-lfs is not installed.
        Example:

        Basic usage example:
        ```py
        trainer = Trainer(...)
        evaluator = Evaluator(...)
        hf_pusher = HFPusher(
            username="chansung",
            access_token=<YOUR-HUGGINGFACE-ACCESS-TOKEN>,
            repo_name="my-model",
            model=trainer.outputs["model"],
            model_blessing=evaluator.outputs["blessing"],
            space_config={
                "app_path": "apps.gradio.semantic_segmentation"
            }
        )
        ```
        """

    pushed_model = types.Channel(type=standard_artifacts.PushedModel)

    spec = HFPusherSpec(
        username=username,
        access_token=access_token,
        repo_name=repo_name,
        space_config=space_config,
        decrypt_fn=decrypt_fn,
        model=model,
        model_blessing=model_blessing,
        pushed_model=pushed_model,
    )

    super().__init__(spec=spec)
