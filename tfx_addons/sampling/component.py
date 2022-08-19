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

from typing import List, Optional

from tfx import types
from tfx.dsl.components.base import base_beam_component, executor_spec
from tfx.types import channel_utils, standard_artifacts
from tfx.utils import json_utils

from tfx_addons.sampling.executor import Executor
from tfx_addons.sampling.spec import SamplerSpec, SamplingStrategy


class Sampler(base_beam_component.BaseBeamComponent):
  """A TFX component to sample examples.

  The sampling component wraps an Apache Beam pipeline to process
  data in an TFX pipeline. This component loads in tf.Record files from
  an earlier example artifact, processes the 'train' split by default,
  samples the split by a given label's classes, and stores the new
  set of sampled examples into its own example artifact in
  tf.Record format.

  The sampling is probabilistic estimation. Note that in small datasets
  this may result in worse datasets or such. This module is meant to
  approximate sampling using probability.

  By default, the component will ignore all examples with a null value
  (more precisely, a value that evaluates to False) for the given label,
  although more values can be added in as necessary. Additionally, it will
  copy all non-'train' splits, through this behavior can be changed as well.
  The component will save the examples in a user-specified number of files,
  and it can be given a name as well.

  ## Example
  ```
  import tfx_addons as tfxa

  under = tfxa.sampling.Sampler(
    examples=example_gen.outputs['examples'],
    sampling_strategy=tfxa.sampling.SamplingStrategy.UNDERSAMPLE
  )
  ```

  Component `outputs` contains:
   - `sampled_examples`: Channel of type `standard_artifacts.Examples` for
      materialized sampled examples, based on the input splits, which includes
      copied splits unless otherwise specified by copy_others.
  """

  SPEC_CLASS = SamplerSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(Executor)

  def __init__(
      self,
      label: str,
      input_data: types.Channel = None,
      splits: Optional[List[str]] = None,
      copy_others: bool = True,
      shards: Optional[int] = 0,
      null_classes: Optional[List[str]] = None,
      sampling_strategy: SamplingStrategy = SamplingStrategy.UNDERSAMPLE):
    """Construct a SamplerComponent.

    Args:
      input_data: A Channel of type `standard_artifacts.Examples`.
      label: The name of the column containing class names to sample by.
      splits: A list containing splits to sample.
      copy_others: Determines whether we copy over the splits that aren't
        sampled, or just exclude them from the output artifact.
      shards: The number of files that each sampled split should
        contain. Default 0 is Beam's tfrecordio function's default.
      null_classes: A list determining which classes that we should not sample.
      sampling_strategy: Wether to undersample or Oversample
    """
    output_data = channel_utils.as_channel([standard_artifacts.Examples()])

    spec = SamplerSpec(
        input_data=input_data,
        output_data=output_data,
        label=label,
        splits=json_utils.dumps(splits),
        copy_others=int(copy_others),
        shards=shards,
        null_classes=json_utils.dumps(null_classes),
        sampling_strategy=sampling_strategy,
    )

    super().__init__(spec=spec)
