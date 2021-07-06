"""Undersampling component definition."""

from typing import Optional, Text, List
from executor import Executor

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.utils import json_utils


UNDERSAMPLER_INPUT_KEY = 'input_data'
UNDERSAMPLER_OUTPUT_KEY = 'output_data'
UNDERSAMPLER_LABEL_KEY = 'label'
UNDERSAMPLER_NAME_KEY = 'name'
UNDERSAMPLER_SPLIT_KEY = 'splits'
UNDERSAMPLER_COPY_KEY = 'copy_others'
UNDERSAMPLER_SHARDS_KEY = 'shards'
UNDERSAMPLER_CLASSES_KEY = 'keep_classes'


class UndersamplerSpec(types.ComponentSpec):
  """Undersampling component spec."""

  PARAMETERS = {
    UNDERSAMPLER_LABEL_KEY: ExecutionParameter(type=str),
    UNDERSAMPLER_NAME_KEY: ExecutionParameter(type=Text, optional=True),
    UNDERSAMPLER_SPLIT_KEY: ExecutionParameter(type=str, optional=True),
    UNDERSAMPLER_COPY_KEY: ExecutionParameter(type=int, optional=True),
    UNDERSAMPLER_SHARDS_KEY: ExecutionParameter(type=int, optional=True),
    UNDERSAMPLER_CLASSES_KEY: ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {
    UNDERSAMPLER_INPUT_KEY: ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
    UNDERSAMPLER_OUTPUT_KEY: ChannelParameter(type=standard_artifacts.Examples),
  }

class Undersampler(base_component.BaseComponent):
  """A TFX component to undersample examples.

  The Undersampling component wraps an Apache Beam pipeline to process
  data in an TFX pipeline. This component loads in tf.Record files from
  an earlier example artifact, processes the 'train' split by default,
  undersamples the split by a given label's classes, and stores the new
  set of undersampled examples into its own example artifact in
  tf.Record format.

  By default, the component will ignore all examples with a null value
  (more precisely, a value that evaluates to False) for the given label,
  although more values can be added in as necessary. Additionally, it will
  copy all non-'train' splits, through this behavior can be changed as well.
  The component will save the examples in a user-specified number of files,
  and it can be given a name as well.

  ## Example
  ```
  # Performs transformations and feature engineering in training and serving.
  under = Undersampler(
    examples=example_gen.outputs['examples'])
  ```

  Component `outputs` contains:
   - `undersampled_examples`: Channel of type `standard_artifacts.Examples` for
                materialized undersampled examples, based on the
                input splits, which includes copied splits unless
                otherwise specified by copy_others.
  """

  SPEC_CLASS = UndersamplerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

  def __init__(
    self,
    label: str,
    input_data: types.Channel = None,
    output_data: types.Channel = None,
    name: Optional[Text] = None,
    splits: Optional[List[Text]] = ["train"],
    copy_others: Optional[bool] = True,
    shards: Optional[int] = 0,
    keep_classes: Optional[List[Text]] = None,
  ):

    """Construct an UndersamplerComponent.
    Args:
      input_data: A Channel of type `standard_artifacts.Examples`.
      output_data: A Channel of type `standard_artifacts.Examples`.
      By default, only the train split is sampled; all others are copied.
      name: Optional unique name. Necessary if multiple components are
      declared in the same pipeline.
      label: The name of the column containing class names to
      undersample by.
      splits: A list containing splits to undersample.
      copy_others: Determines whether we copy over the splits that aren't
      undersampled, or just exclude them from the output artifact.
      shards: The number of files that each undersampled split should
      contain. Default 0 is Beam's tfrecordio function's default.
      keep_classes: A list determining which classes that we should s
      not undersample.
    """

    if not output_data:
      output_data = channel_utils.as_channel([standard_artifacts.Examples()])

    spec = UndersamplerSpec(
      input_data=input_data,
      output_data=output_data,
      label=label,
      name=name,
      splits=json_utils.dumps(splits),
      copy_others=int(copy_others),
      shards=shards,
      keep_classes=json_utils.dumps(keep_classes),
    )

    super().__init__(spec=spec)
