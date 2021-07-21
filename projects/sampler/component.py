"""Sampling component definition."""

from typing import Optional, Text, List
from sampler.executor import Executor

from tfx import types
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import executor_spec
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.utils import json_utils

class SamplerSpec(types.ComponentSpec):
  """Sampling component spec."""

  PARAMETERS = {
    'label': ExecutionParameter(type=str),
    'name': ExecutionParameter(type=Text, optional=True),
    'splits': ExecutionParameter(type=str, optional=True),
    'copy_others': ExecutionParameter(type=int, optional=True),
    'shards': ExecutionParameter(type=int, optional=True),
    'keep_classes': ExecutionParameter(type=str, optional=True),
    'undersample': ExecutionParameter(type=int, optional=True)
  }
  INPUTS = {
    'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
    'output_data': ChannelParameter(type=standard_artifacts.Examples),
  }

class Sampler(base_beam_component.BaseBeamComponent):
  """A TFX component to sample examples.

  The sampling component wraps an Apache Beam pipeline to process
  data in an TFX pipeline. This component loads in tf.Record files from
  an earlier example artifact, processes the 'train' split by default,
  samples the split by a given label's classes, and stores the new
  set of sampled examples into its own example artifact in
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
  under = Sampler(
    examples=example_gen.outputs['examples'])
  ```

  Component `outputs` contains:
   - `sampled_examples`: Channel of type `standard_artifacts.Examples` for
                materialized sampled examples, based on the
                input splits, which includes copied splits unless
                otherwise specified by copy_others.
  """

  SPEC_CLASS = SamplerSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(Executor)

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
    undersample: bool = True
  ):

    """Construct an SamplerComponent.
    Args:
      input_data: A Channel of type `standard_artifacts.Examples`.
      output_data: A Channel of type `standard_artifacts.Examples`.
      By default, only the train split is sampled; all others are copied.
      name: Optional unique name. Necessary if multiple components are
      declared in the same pipeline.
      label: The name of the column containing class names to
      sample by.
      splits: A list containing splits to sample.
      copy_others: Determines whether we copy over the splits that aren't
      sampled, or just exclude them from the output artifact.
      shards: The number of files that each sampled split should
      contain. Default 0 is Beam's tfrecordio function's default.
      keep_classes: A list determining which classes that we should s
      not sample.
    """

    if not output_data:
      output_data = channel_utils.as_channel([standard_artifacts.Examples()])

    spec = SamplerSpec(
      input_data=input_data,
      output_data=output_data,
      label=label,
      name=name,
      splits=json_utils.dumps(splits),
      copy_others=int(copy_others),
      shards=shards,
      keep_classes=json_utils.dumps(keep_classes),
     undersample=int(undersample)
    )

    super().__init__(spec=spec)
