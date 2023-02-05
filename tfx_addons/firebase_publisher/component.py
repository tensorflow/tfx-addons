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
    """The FirebasePublisher TFX component.
       FirebasePublisher deploys a trained or blessed model to Firebase ML.
       This is designed to work as a downstream component of Trainer and
       Evaluator(optional) components. Trainer gives trained model, and
       Evaluator gives information whether the trained model is blessed or
       not after evaluation of the model. This component only publishes a
       model when it is blessed. If Evaluator is not specified, the input
       model will always be published.
       **Important Note:** Firebase ML allows to host a model under 40mb.
       if the model size exceeds 40mb, RuntimError will be raised.


    Args:
      display_name: name to identify a hosted model in Firebase ML. this should
        be a unique value because it will be used to search a existing model to
        update.
      storage_bucket: GCS bucket where the hosted model will be stored.
      app_name: the name of Firebase app to determine the scope. if not given,
        default app name, '[DEFAULT]' will be used.
      tags: tags to be attached to the hosted ML model. any meaningful tags will
        be helpful to understand your model in Firebase ML platform.
      options: additional configurations to be passed to initialize Firebase
        app. refer to the official document to find out which options are
        available at [`initialize_app()`]
        (https://firebase.google.com/docs/reference/admin/python/firebase_admin#initialize_app).
      credential_path: location of GCS or local file system where the Service
        Account(SA) Key file is. the SA should have sufficeint permissions to
        create and view hosted models in Firebase ML. If this parameter is not
        given, Application Default Credentials will be used in GCP environment.
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
      RuntimeError: when model size exceeds 40mb.
    Example:
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
