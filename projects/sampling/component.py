from typing import Optional, Text, List

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from executor import UndersamplingExecutor
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter


class UndersamplingComponentSpec(types.ComponentSpec):

  PARAMETERS = {
      'name': ExecutionParameter(type=Text, optional=True),
      'splits': ExecutionParameter(type=List[Text], optional=True)
  }
  INPUTS = {
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      'output_data': ChannelParameter(type=standard_artifacts.Examples),
  }


class UndersamplingComponent(base_component.BaseComponent):
  SPEC_CLASS = UndersamplingComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(UndersamplingExecutor)

  def __init__(self,
               input_data: types.Channel = None,
               output_data: types.Channel = None,
               name: Optional[Text] = None,
               splits: Optional[List[Text]] = ['train']):

    """Construct an UndersamplingComponent.
    Args:
      input_data: A Channel of type `standard_artifacts.Examples`.
      output_data: A Channel of type `standard_artifacts.Examples`.
        By default, only the train split is sampled; all others are copied.
      name: Optional unique name. Necessary if multiple components are
        declared in the same pipeline.
    """

    if not output_data:
      output_data = channel_utils.as_channel([standard_artifacts.Examples()])

    spec = UndersamplingComponentSpec(input_data=input_data, output_data=output_data, name=name, splits=splits)
    super(UndersamplingComponent, self).__init__(spec=spec)
