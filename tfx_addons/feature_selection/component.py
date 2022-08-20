# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Feature Selection component for tfx_addons"""
import importlib
import json
import os

import tensorflow as tf
from tfx.dsl.component.experimental.decorators import component
from tfx.types import artifact, artifact_utils
from tfx.types.standard_artifacts import Examples
from tfx.v1.dsl.components import InputArtifact, OutputArtifact, Parameter
from tfx_bsl.coders import example_coder


# Output artifact containing required data related to feature selection
class FeatureSelectionArtifact(artifact.Artifact):
  """Output artifact containing feature scores from the Feature Selection component"""
  TYPE_NAME = 'Feature Selection'
  PROPERTIES = {
      'scores': artifact.Property(type=artifact.PropertyType.JSON_VALUE),
      'p_values': artifact.Property(type=artifact.PropertyType.JSON_VALUE),
      'selected_features':
      artifact.Property(type=artifact.PropertyType.JSON_VALUE)
  }


# reads and returns data from TFRecords at URI as a list of dictionaries with values as numpy arrays
def _get_data_from_tfrecords(train_uri):
  # get all the data files
  train_uri = [
      os.path.join(train_uri, file_path)
      for file_path in _get_file_list(train_uri)
  ]
  raw_dataset = tf.data.TFRecordDataset(train_uri, compression_type='GZIP')

  np_dataset = []
  for tfrecord in raw_dataset:
    serialized_example = tfrecord.numpy()
    example = example_coder.ExampleToNumpyDict(serialized_example)
    np_dataset.append(example)

  return np_dataset


# returns data in list and nested list formats compatible with sklearn
def _data_preprocessing(np_dataset, target_feature):

  # getting the required data without any metadata
  np_dataset = [{k: v[0]
                 for k, v in example.items()} for example in np_dataset]

  # extracting `y`
  target = [i.pop(target_feature) for i in np_dataset]
  feature_keys = list(np_dataset[0].keys())
  # getting `X`
  input_data = [[i[j] for j in feature_keys] for i in np_dataset]

  return [feature_keys, target, input_data]


# update example with selected features
def _update_example(selected_features, orig_example):
  result = {}
  for key, feature in orig_example.features.feature.items():
    if key in selected_features:
      result[key] = feature

  new_example = tf.train.Example(features=tf.train.Features(feature=result))
  return new_example


# get a list of files for the specified path
def _get_file_list(dir_path):
  file_list = [
      f for f in os.listdir(dir_path)
      if os.path.isfile(os.path.join(dir_path, f))
  ]
  return file_list


@component
def FeatureSelection(  # pylint: disable=C0103
    orig_examples: InputArtifact[Examples],
    feature_selection: OutputArtifact[FeatureSelectionArtifact],
    updated_data: OutputArtifact[Examples],
    module_file: Parameter[str] = None,
    module_path: Parameter[str] = None,
):
  """Runs a user-specified feature selection algorithm on an `Examples` artifact
    Args:
      - orig_examples: An `Examples` input artifact with the data to be
        processed
      - module_file: Python module file containing the configuration
        Example: `modules_files.module_file_a`
        Exactly one of `module_file` and `module_path` should be passed.
        If both are used, module_file would be preferred
      - module_path: Python module path containing the configuration
        Example: `absolute_path/module_files/module_file_a.py` or
        `./module_files/module_file_a.py`
        Exactly one of `module_file` and `module_path` should be passed.
        If both are used, module_file would be preferred

    Module file configuration:
    - SELECTOR_PARAMS: Parameters for SelectorFunc in the form of
      a kwargs dictionary
      Example: {"score_func": chi2, "k": 2}
      Here, `chi2` has been imported from sklearn.feature_selection
    - TARGET_FEATURE: Name of the feature containing target data
    - SelectorFunc: Selector function for univariate feature selection
      Example: SelectKBest, SelectPercentile from sklearn.feature_selection
  """

  # importing the required functions and variables from the module file

  if module_file:
    modules = importlib.import_module(module_file)
  elif module_path:
    module_spec = importlib.util.spec_from_file_location(
        "all_modules", module_path)
    modules = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(modules)

  mod_names = ["SELECTOR_PARAMS", "TARGET_FEATURE", "SelectorFunc"]
  selector_params, target_feature, selector_func = [
      getattr(modules, i) for i in mod_names
  ]

  # uri for the required data
  train_uri = artifact_utils.get_split_uri([orig_examples], 'train')
  np_dataset = _get_data_from_tfrecords(train_uri)
  feature_keys, target_data, input_data = _data_preprocessing(
      np_dataset, target_feature)

  # Select features based on scores
  selector = selector_func(**selector_params)
  selector.fit_transform(input_data, target_data)

  # adding basic info to the updated example artifact as output
  updated_data.split_names = orig_examples.split_names
  updated_data.span = orig_examples.span

  # generate a list of selected features by matching FEATURE_KEYS to selected indices
  selected_features = [
      val for (idx, val) in enumerate(feature_keys)
      if idx in selector.get_support(indices=True)
  ]

  # convert string to array
  split_arr = json.loads(orig_examples.split_names)

  # update feature per split
  for split in split_arr:
    split_uri = artifact_utils.get_split_uri([orig_examples], split)
    new_split_uri = artifact_utils.get_split_uri([updated_data], split)
    os.mkdir(new_split_uri)

    for file in _get_file_list(split_uri):
      split_dataset = tf.data.TFRecordDataset(os.path.join(split_uri, file),
                                              compression_type='GZIP')

      # write the TFRecord
      with tf.io.TFRecordWriter(path=os.path.join(new_split_uri, file),
                                options="GZIP") as writer:
        for split_record in split_dataset.as_numpy_iterator():
          example = tf.train.Example()
          example.ParseFromString(split_record)

          updated_example = _update_example(selected_features, example)
          writer.write(updated_example.SerializeToString())

  # get scores and p-values for artifacts
  selector_scores = selector.scores_
  selector_p_values = selector.pvalues_

  # merge scores and pvalues with feature keys to create a dictionary
  selector_scores_dict = dict(zip(feature_keys, selector_scores))
  selector_pvalues_dict = dict(zip(feature_keys, selector_p_values))

  # populate artifact with the required properties
  feature_selection.scores = selector_scores_dict
  feature_selection.p_values = selector_pvalues_dict
  feature_selection.selected_features = selected_features
