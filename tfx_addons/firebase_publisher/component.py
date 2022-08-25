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
  """Component for pushing model to Firebase ML"""

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
