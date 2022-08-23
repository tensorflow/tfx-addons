import time
from typing import Any, Dict, List, Optional

from absl import logging
from tfx import types
from tfx.types import artifact_utils, standard_component_specs

from tfx.components.pusher import executor as tfx_pusher_executor
from tfx_addons.firebase_publisher import runner

_DISPLAY_NAME_KEY = "display_name"
_STORAGE_BUCKET_KEY = "storage_bucket"
_TAGS_KEY = "tags"
_OPTIONS_KEY = "options"
_CREDENTIAL_PATH_KEY = "credential_path"

class Executor(tfx_pusher_executor.Executor):
  """Push a model to Firebase ML"""

  def Do(
      self,
      input_dict: Dict[str, List[types.Artifact]],
      output_dict: Dict[str, List[types.Artifact]],
      exec_properties: Dict[str, Any],
  ):
    """Overrides the tfx_pusher_executor.
    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from evaluator.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_push: A list of 'ModelPushPath' artifact of size one. It will
          include the model in this push execution if the model was pushed.
      exec_properties: ...
    Raises:
      ValueError: ...
      RuntimeError: if the job fails.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    model_push = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.PUSHED_MODEL_KEY]
    )
    if not self.CheckBlessing(input_dict):
      self._MarkNotPushed(model_push)
      return
    model_path = self.GetModelPath(input_dict)
    model_version_name = f"v{int(time.time())}"

    pushed_model_path = runner.deploy_model_for_firebase_ml(
      display_name=exec_properties.get(_DISPLAY_NAME_KEY),
      storage_bucket=exec_properties.get(_STORAGE_BUCKET_KEY),
      credential_path=exec_properties.get(_CREDENTIAL_PATH_KEY, None),
      tags=exec_properties.get(_TAGS_KEY, []),
      options=exec_properties.get(_OPTIONS_KEY, {}),
      model_path=model_path,      
      model_version=model_version_name,
    )

    self._MarkPushed(
      model_push, 
      pushed_destination=pushed_model_path)
