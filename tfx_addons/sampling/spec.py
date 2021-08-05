# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Sampling component definition."""

import enum
from typing import Text

from tfx import types
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter

SAMPLER_INPUT_KEY = 'input_data'
SAMPLER_OUTPUT_KEY = 'output_data'
SAMPLER_LABEL_KEY = 'label'
SAMPLER_NAME_KEY = 'name'
SAMPLER_SPLIT_KEY = 'splits'
SAMPLER_COPY_KEY = 'copy_others'
SAMPLER_SHARDS_KEY = 'shards'
SAMPLER_CLASSES_KEY = 'null_classes'
SAMPLER_SAMPLE_KEY = 'sampling_strategy'


class SamplingStrategy(enum.IntEnum):
  """Determines which kind of sampling to perform."""
  UNDERSAMPLE = 1
  OVERSAMPLE = 2


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
