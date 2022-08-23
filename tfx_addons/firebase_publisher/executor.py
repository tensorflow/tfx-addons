import time
from typing import Any, Dict, List

from tfx import types
from tfx.types import artifact_utils, standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx.utils import name_utils
from tfx.utils import telemetry_utils
from tfx.dsl.io import fileio

from tfx.components.pusher import executor as tfx_pusher_executor
from tfx_addons.firebase_publisher import runner

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

        custom_config = json_utils.loads(
            exec_properties.get(_CUSTOM_CONFIG_KEY, "null")
        )

        if custom_config is not None and not isinstance(custom_config, Dict):
            raise ValueError(
                "custom_config in execution properties needs to be a dict."
            )

        firebase_ml_args = custom_config.get(constants.FIREBASE_ML_KEY)
        if not firebase_ml_args:
            raise ValueError("'FIREBASE_ML' is missing in 'custom_config'")
        model_push = artifact_utils.get_single_instance(
            output_dict[standard_component_specs.PUSHED_MODEL_KEY]
        )
        if not self.CheckBlessing(input_dict):
            self._MarkNotPushed(model_push)
            return

        # Deploy the model.
        io_utils.copy_dir(src=self.GetModelPath(input_dict), dst=model_push.uri)
        model_path = model_push.uri

        executor_class_path = name_utils.get_full_name(self.__class__)
        with telemetry_utils.scoped_labels(
            {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}
        ):
            job_labels = telemetry_utils.make_labels_dict()

        model_name = f"v{int(time.time())}"
        pushed_model_path = runner.deploy_model_for_firebase_ml(
            model_version_name=model_name,
            model_path=model_path,
            firebase_ml_args=firebase_ml_args,
        )

        self._MarkPushed(model_push, pushed_destination=pushed_model_path)