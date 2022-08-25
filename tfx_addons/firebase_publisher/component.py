"""Firebase Publisher TFX Component.

The FirebasePublisher is used to deploy model to Firebase ML.
"""

from typing import Dict, List, Optional

from pipeline.components.pusher.FirebasePublisher import executor
from tfx import types
from tfx.dsl.components.base import base_component, executor_spec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter

MODEL_KEY = "model"
PUSHED_MODEL_KEY = "pushed_model"
MODEL_BLESSING_KEY = "model_blessing"


class FirebasePublisherSpec(types.ComponentSpec):
  """ComponentSpec for TFX FirebasePublisher Component."""

  PARAMETERS = {
      "app_name": ExecutionParameter(type=str),
      "display_name": ExecutionParameter(type=str),
      "storage_bucket": ExecutionParameter(type=str),
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
  """Component for pushing model to Firebase ML

  The `FirebasePublisher` is a [TFX
  Component](https://www.tensorflow.org/tfx/guide/understanding_tfx_pipelines#component)
  that deploy a ML model to Firebase ML.

  This component deploys(publishes) a model from the upstream component
  such as [`Trainer`](https://www.tensorflow.org/tfx/guide/trainer) to
  Firebase ML which hosts ML models that mobile applications can download.
  To find more about Firebase ML, please refer to the official [webpage](
  https://firebase.google.com/products/ml).

  it only publishes a model when it is blessed, and the blessness is
  determined by the upstream Evaluator component. If Evaluator is not
  specified, the input model will always published.

  Basic usage example:
  ```py
  trainer = Trainer(...)
  evaluator = Evaluator(...)

  fb_publisher = FirebasePublisher(
    display_name="model_on_firebase",
    storage_bucket="firebase_ml", # only the bucket name without gs://
    credential_path="gs://....json",
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
    """Construct a FirebasePublisher component.

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
