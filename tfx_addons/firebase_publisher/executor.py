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
    """Overrides the tfx_pusher_executor to leverage some of utility costs methods

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from evaluator.
      output_dict: Output dict from key to a list of artifacts, including:
        - pushed_model: A 'PushedModel' artifact. It will contain a pushed
          destination information.
      exec_properties: An optional dict of execution properties, including:
        - display_name: display name to appear in Firebase ML. this should be
          a unique value since it will be used to search a existing model to
          update.
        - storage_bucket: GCS bucket where the hosted model is stored.
        - app_name: the name of Firebase app to determin the scope.
        - tags: tags to be attached to the hosted ML model.
        - credential_path: an optional parameter, and it indicates GCS or local
          location where a Service Account Key file is stored. If this parameter
          is not given, Application Default Credentials will be used in GCP
          environment.
        - options: additional configurations to be passed to initialize Firebase
          app. refer to the official document about the [`initialize_app()`](
          https://firebase.google.com/docs/reference/admin/python/firebase_admin#initialize_app).

    Raises:
      ValueError: when the each item in tags has disallowed characters.
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
