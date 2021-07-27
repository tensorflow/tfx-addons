"""Sampling component definition."""

from typing import Text

from tfx import types
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter

SAMPLER_INPUT_KEY = 'input_data'
SAMPLER_OUTPUT_KEY = 'output_data'
SAMPLER_LABEL_KEY = 'label'
SAMPLER_NAME_KEY = 'name'
SAMPLER_SPLIT_KEY = 'splits'
SAMPLER_COPY_KEY = 'copy_others'
SAMPLER_SHARDS_KEY = 'shards'
SAMPLER_CLASSES_KEY = 'keep_classes'
SAMPLER_SAMPLE_KEY = 'undersample'

class SamplerSpec(types.ComponentSpec):
  """Sampling component spec."""

  PARAMETERS = {
    SAMPLER_LABEL_KEY: ExecutionParameter(type=str),
    SAMPLER_NAME_KEY: ExecutionParameter(type=Text, optional=True),
    SAMPLER_SPLIT_KEY: ExecutionParameter(type=str, optional=True),
    SAMPLER_COPY_KEY: ExecutionParameter(type=int, optional=True),
    SAMPLER_SHARDS_KEY: ExecutionParameter(type=int, optional=True),
    SAMPLER_CLASSES_KEY: ExecutionParameter(type=str, optional=True),
    SAMPLER_SAMPLE_KEY: ExecutionParameter(type=int, optional=True)
  }
  INPUTS = {
    SAMPLER_INPUT_KEY: ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
    SAMPLER_OUTPUT_KEY: ChannelParameter(type=standard_artifacts.Examples),
  }
