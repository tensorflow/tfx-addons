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
"""Sampling pipeline example using TFX."""

from __future__ import absolute_import, division, print_function

import os
from typing import List, Text

import absl
import tensorflow_model_analysis as tfma
from tfx.components import (CsvExampleGen, Evaluator, ExampleValidator, Pusher,
                            SchemaGen, StatisticsGen, Trainer, Transform)
from tfx.components.trainer.executor import Executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import (latest_artifacts_resolver,
                                  latest_blessed_model_resolver)
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

from tfx_addons.sampling.component import Sampler

_pipeline_name = 'sampling_credit_card'
_sampling_root = os.path.dirname(__file__)
_data_root = os.path.join(_sampling_root, 'data')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_sampling_root, 'sampler_utils.py')
_serving_model_dir = os.path.join(_sampling_root, 'serving_model',
                                  _pipeline_name)
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     module_file: Text, serving_model_dir: Text,
                     metadata_path: Text,
                     beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
  """Implements an example pipeline with the sampling component witin TFX."""

  example_gen = CsvExampleGen(input_base=data_root)
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
                         infer_feature_shape=False)
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Sampler component, with input examples and class label.
  sample = Sampler(input_data=example_gen.outputs['examples'],
                   splits=['train'],
                   label='Class',
                   shards=10)

  transform = Transform(examples=sample.outputs['output_data'],
                        schema=schema_gen.outputs['schema'],
                        module_file=module_file)
  latest_model_resolver = resolver.Resolver(
      strategy_class=latest_artifacts_resolver.LatestArtifactsResolver,
      latest_model=Channel(type=Model)).with_id('latest_model_resolver')
  trainer = Trainer(
      module_file=module_file,
      custom_executor_spec=executor_spec.ExecutorClassSpec(Executor),
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      base_model=latest_model_resolver.outputs['latest_model'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))
  model_resolver = resolver.Resolver(
      strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(
          type=ModelBlessing)).with_id('latest_blessed_model_resolver')

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name='eval')],
      slicing_specs=[
          tfma.SlicingSpec(),
          tfma.SlicingSpec(feature_keys=['trip_start_hour'])
      ],
      metrics_specs=[
          tfma.MetricsSpec(
              thresholds={
                  'accuracy':
                  tfma.config.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.6}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10}))
              })
      ])

  evaluator = Evaluator(examples=example_gen.outputs['examples'],
                        model=trainer.outputs['model'],
                        baseline_model=model_resolver.outputs['model'],
                        eval_config=eval_config)
  pusher = Pusher(model=trainer.outputs['model'],
                  model_blessing=evaluator.outputs['blessing'],
                  push_destination=pusher_pb2.PushDestination(
                      filesystem=pusher_pb2.PushDestination.Filesystem(
                          base_directory=serving_model_dir)))

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen,
          example_validator,
          sample,
          transform,
          latest_model_resolver,
          trainer,
          model_resolver,
          evaluator,
          pusher,
      ],
      enable_cache=False,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $python sampler_pipeline_local.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)

  LocalDagRunner().run(
      _create_pipeline(pipeline_name=_pipeline_name,
                       pipeline_root=_pipeline_root,
                       data_root=_data_root,
                       module_file=_module_file,
                       serving_model_dir=_serving_model_dir,
                       metadata_path=_metadata_path,
                       beam_pipeline_args=_beam_pipeline_args))
