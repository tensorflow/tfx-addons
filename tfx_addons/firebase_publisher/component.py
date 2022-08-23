from typing import Any, Dict, Optional

from tfx import types
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter
from tfx.dsl.components.base import executor_spec

from tfx_addons.firebase_publisher import executor

MODEL_KEY = 'model'
PUSHED_MODEL_KEY = 'pushed_model'
MODEL_BLESSING_KEY = 'model_blessing'

class FirebasePublisherSpec(types.ComponentSpec):
  """ComponentSpec for TFX FirebasePublisher Component."""

  PARAMETERS = {
      'display_name': ExecutionParameter(type=str),
      'storage_bucket': ExecutionParameter(type=str),
      'tags': ExecutionParameter(type=List, optional=True),
      'options': ExecutionParameter(type=Dict, optional=True),
      'credential_path': ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
      MODEL_KEY:
          ChannelParameter(type=standard_artifacts.Model, optional=True),
      MODEL_BLESSING_KEY:
          ChannelParameter(
              type=standard_artifacts.ModelBlessing, optional=True),
  }
  OUTPUTS = {
      PUSHED_MODEL_KEY: ChannelParameter(type=standard_artifacts.PushedModel),
  }

class FirebasePublisher(types.ComponentSpec):
    """Component for pushing model to Firebase ML"""

    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(
        self,
        display_name: str,
        storage_bucket: str,
        tags: Optional[List[str]] = None,
        options: Optional[Dict] = None,
        credential_path: Optional[str] = None,
        model: Optional[types.BaseChannel] = None,
        model_blessing: Optional[types.BaseChannel] = None,        
    ):
        """Construct a FirebasePublisher component.
        Args:
            model : the model from the upstream TFX component such as Trainer
            model_blessing : the output of blessing from the Evaluator component 
                to indicate if the given model is good enough to be pushed
            display_name : display name to appear in Firebase ML. 
                This should be a unique value since it will be used to search a existing model to update
            tags : tags to appear in Firebase ML
            storage_bucket : GCS bucket where the hosted model is stored. gs:// should not be included
            credential_path : an optional parameter, and it indicates GCS or 
                local location where a Service Account Key (JSON) file is stored. 
                If this parameter is not given, Application Default Credentials will be used in GCP environment
            options : additional configurations to be passed to initialize Firebase app        
        """
        super().__init__(
            display_name=display_name,
            tags=tags,
            storage_bucket=storage_bucket,
            options=options,
            credential_path=credential_path,
            model=model,
            model_blessing=model_blessing,            
        )