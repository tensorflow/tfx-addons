from typing import Any, Dict, Optional

from tfx import types
from tfx.components.pusher import component as pusher_component
from tfx.dsl.components.base import executor_spec

from tfx_addons.firebase_publisher import executor

class FirebasePublisher(pusher_component.Pusher):
    """Component for pushing model to Firebase ML"""

    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(
        self,
        display_name: str,
        tags: List[str],
        storage_bucket: str,
        model: Optional[types.BaseChannel] = None,
        model_blessing: Optional[types.BaseChannel] = None,        
        options: Optional[Dict] = None,
        credential_path: Optional[str] = None,
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
            model=model,
            model_blessing=model_blessing,            
            options=options,
            credential_path=credential_path,
        )