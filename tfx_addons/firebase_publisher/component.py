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
"""Firebase Publisher TFX Component.

The FirebasePublisher is used to deploy model to Firebase ML.
"""

from typing import Dict, List, Optional

from tfx import types
from tfx.dsl.components.base import base_component, executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter

from tfx_addons.firebase_publisher import executor

MODEL_KEY = "model"
PUSHED_MODEL_KEY = "pushed_model"
MODEL_BLESSING_KEY = "model_blessing"


class FirebasePublisherSpec(types.ComponentSpec):
  """ComponentSpec for TFX FirebasePublisher Component."""

  PARAMETERS = {
      "display_name": ExecutionParameter(type=str),
      "storage_bucket": ExecutionParameter(type=str),
      "app_name": ExecutionParameter(type=str, optional=True),
      "tags": ExecutionParameter(type=List, optional=True),
      "options": ExecutionParameter(type=Dict, optional=True),
      "credential_path": ExecutionParameter(type=str, optional=True),
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


class FirebasePublisher(base_component.BaseComponent):
  """Component for pushing model to Firebase ML.

  The `FirebasePublisher` is a [TFX
  Component](https://www.tensorflow.org/tfx/guide/understanding_tfx_pipelines#component)
  that pushes a model to Firebase ML.
  This component publishes a model from an upstream component
  such as [`Trainer`](https://www.tensorflow.org/tfx/guide/trainer) to
  Firebase ML that hosts ML models that mobile applications can consume.
  As such, it provides better interoperability between ML models and mobile
  applications that need to consume them. To find more about Firebase ML,
  please refer to the official
  [webpage](https://firebase.google.com/products/ml).

  It only publishes a model when it is blessed, and the "blessness" is
  determined by the upstream `Evaluator` component. If Evaluator is not
  specified, the input model will always be published.

  Basic usage example:
  ```py
  trainer = Trainer(...)
  evaluator = Evaluator(...)

  fb_publisher = FirebasePublisher(
    display_name="model_on_firebase",
    storage_bucket="firebase_ml", # only the bucket name without gs://
    credential_path="gs://abc.json",
    model=trainer.outputs["model"],
    model_blessing=evaluator.outputs["blessing"]
  )
  ```
  """

  SPEC_CLASS = FirebasePublisherSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      display_name: str,
      storage_bucket: str,
      app_name: Optional[str] = None,
      tags: Optional[List[str]] = None,
      options: Optional[Dict] = None,
      credential_path: Optional[str] = None,
      model: Optional[types.BaseChannel] = None,
      model_blessing: Optional[types.BaseChannel] = None,
  ):
    pushed_model = types.Channel(type=standard_artifacts.PushedModel)

    spec = FirebasePublisherSpec(app_name=app_name,
                                 display_name=display_name,
                                 tags=tags,
                                 storage_bucket=storage_bucket,
                                 options=options,
                                 credential_path=credential_path,
                                 model=model,
                                 model_blessing=model_blessing,
                                 pushed_model=pushed_model)

    super().__init__(spec=spec)
