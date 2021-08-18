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
"""Penguin example using TFX and XGBoost.

Run pipeline:
$ python examples/xgboost_penguins/penguin_pipeline_local.py
"""

import os
from pathlib import Path
from typing import List, Text

from absl import logging
from tfx import v1 as tfx

_pipeline_name = 'penguin_xgboost_local'

# Feel free to customize as needed.
_root = Path(__file__).parent
_data_root = str(_root / 'data')

# Python module file to inject customized logic into TFX components.
_module_file = str(_root / 'utils.py')

# Pipeline artifact locations. You can store these files anywhere on your local filesystem.
_tfx_root = Path(os.environ['HOME']) / 'tfx'
_pipeline_root = str(_tfx_root / 'pipelines' / _pipeline_name)
_metadata_path = str(_tfx_root / 'metadata' / _pipeline_name / 'metadata.db')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_threading',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=1',
]


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_root: Text,
    module_file: Text,
    metadata_path: Text,
    beam_pipeline_args: List[Text],
) -> tfx.dsl.Pipeline:
  """Create a TFX logical pipeline.

  Args:
      pipeline_name: name of the pipeline
      pipeline_root: directory to store pipeline artifacts
      data_root: input data directory
      module_file: module file to inject customized logic into TFX components
      metadata_path: path to local sqlite database file
      beam_pipeline_args: arguments for Beam powered components
  """
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = tfx.components.SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = tfx.components.ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'],
  )

  trainer_custom_config = {
      'objective': 'reg:squarederror',
      'learning_rate': 0.3,
      'max_depth': 4,
      'num_boost_round': 200,
      'early_stopping_rounds': 40,
  }

  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      train_args=tfx.proto.TrainArgs(),
      eval_args=tfx.proto.EvalArgs(),
      custom_config=trainer_custom_config,
  )

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen,
          example_validator,
          trainer,
      ],
      enable_cache=True,
      metadata_connection_config=tfx.orchestration.metadata.
      sqlite_metadata_connection_config(metadata_path),
      beam_pipeline_args=beam_pipeline_args,
  )


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tfx.orchestration.LocalDagRunner().run(
      create_pipeline(pipeline_name=_pipeline_name,
                      pipeline_root=_pipeline_root,
                      data_root=_data_root,
                      module_file=_module_file,
                      metadata_path=_metadata_path,
                      beam_pipeline_args=_beam_pipeline_args))
