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
"""Python module file with Penguin pipeline functions and necessary utils.

Include entry point run_fn() to be called by TFX Trainer.
"""

import os
import tempfile
from typing import Text, Tuple

import absl
import numpy as np
import xgboost as xgb
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx.dsl.io import fileio
from tfx.utils import io_utils
from tfx_bsl.tfxio import dataset_options

_FEATURE_KEYS = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]
_LABEL_KEY = 'species'

# The Penguin dataset has 342 records, and is divided into train and eval
# splits in a 2:1 ratio.
_TRAIN_DATA_SIZE = 228
_TRAIN_BATCH_SIZE = 20


def _input_fn(
    file_pattern: Text,
    data_accessor: DataAccessor,
    schema: schema_pb2.Schema,
    batch_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.
    batch_size: the number of records to combine in each batch.

  Returns:
    A (features, indices) tuple where features is a matrix of features, and
      indices is a single vector of label indices.
  """
  record_batch_iterator = data_accessor.record_batch_factory(
      file_pattern,
      dataset_options.RecordBatchesOptions(batch_size=batch_size,
                                           num_epochs=1), schema)

  feature_list = []
  label_list = []
  for record_batch in record_batch_iterator:
    record_dict = {}
    for column, field in zip(record_batch, record_batch.schema):
      record_dict[field.name] = column.flatten()

    label_list.append(record_dict[_LABEL_KEY])
    features = [record_dict[key] for key in _FEATURE_KEYS]
    feature_list.append(np.stack(features, axis=-1))

  return np.concatenate(feature_list), np.concatenate(label_list)


def _train(fn_args, x_train, y_train, x_eval, y_eval):
  params = {
      'objective': fn_args.custom_config['objective'],
      'learning_rate': fn_args.custom_config['learning_rate'],
      'max_depth': fn_args.custom_config['max_depth']
  }
  matrix_train = xgb.DMatrix(x_train, label=y_train)
  matrix_eval = xgb.DMatrix(x_eval, label=y_eval)
  model = xgb.train(
      params=params,
      dtrain=matrix_train,
      num_boost_round=fn_args.custom_config['num_boost_round'],
      early_stopping_rounds=fn_args.custom_config['early_stopping_rounds'],
      evals=[(matrix_eval, 'eval')])
  return model


def _save_model(output_dir, model):
  fileio.mkdir(output_dir)
  with fileio.open(os.path.join(output_dir, 'model_dump.json'), 'w') as fout:
    model.dump_model(fout, with_stats=True, dump_format='json')
  with tempfile.NamedTemporaryFile() as fout:
    model.save_model(fout.name)
    fileio.copy(fout.name, os.path.join(output_dir, 'model.bst'))

  # If the given file name has `.json` suffix, xgboost saves the model in json format.
  with tempfile.NamedTemporaryFile(suffix='.json') as fout:
    model.save_model(fout.name)
    fileio.copy(fout.name, os.path.join(output_dir, 'model.json'))


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  x_train, y_train = _input_fn(fn_args.train_files, fn_args.data_accessor,
                               schema)
  x_eval, y_eval = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema)
  model = _train(fn_args, x_train, y_train, x_eval, y_eval)
  absl.logging.info(model)
  _save_model(fn_args.serving_model_dir, model)
