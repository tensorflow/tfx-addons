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

import importlib
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from tfx_bsl.coders import example_coder
from tfx.dsl.component.experimental.decorators import component
from tfx.types import artifact, artifact_utils
from tfx.types.standard_artifacts import Examples
from tfx.v1.dsl.components import OutputArtifact, InputArtifact, Parameter


# Output artifact containing required data related to feature selection
"""Custom Artifact type"""
class FeatureSelectionArtifact(artifact.Artifact):
  """Output artifact containing feature scores from the Feature Selection component"""
  TYPE_NAME = 'Feature Selection'
  PROPERTIES = {
    'scores': artifact.Property(type=artifact.PropertyType.JSON_VALUE),
    'p_values': artifact.Property(type=artifact.PropertyType.JSON_VALUE),
    'selected_features': artifact.Property(type=artifact.PropertyType.JSON_VALUE)
  }


# reads and returns data from TFRecords at URI as a list of dictionaries with values as numpy arrays
def get_data_from_TFRecords(train_uri):
  train_uri = [os.path.join(train_uri, 'data_tfrecord-00000-of-00001.gz')]
  raw_dataset = tf.data.TFRecordDataset(train_uri, compression_type='GZIP')

  np_dataset = []
  for tfrecord in raw_dataset:
    serialized_example = tfrecord.numpy()
    example = example_coder.ExampleToNumpyDict(serialized_example)
    np_dataset.append(example)

  return np_dataset


# returns data in list and nested list formats compatible with sklearn
def data_preprocessing(np_dataset, target_feature):

  # getting the required data without any metadata
  np_dataset = [{k: v[0] for k, v in example.items()} for example in np_dataset]

  # extracting `y`
  target = [i.pop(target_feature) for i in np_dataset]
  feature_keys = list(np_dataset[0].keys())
  # getting `X`
  input_data = [[i[j] for j in feature_keys] for i in np_dataset]

  return [feature_keys, target, input_data]


# update example with selected features
def update_example(selected_features, orig_example):
  result = {}
  for key, feature in orig_example.features.feature.items():
    if key in selected_features:
      result[key] = feature

  new_example = tf.train.Example(features=tf.train.Features(feature=result))
  return new_example

# get a list of files for the specifiEd path
def get_file_list(dir_path):
  file_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
  return file_list

"""
Feature selection component
"""
@component
def FeatureSelection(module_file: Parameter[str],
  orig_examples: InputArtifact[Examples],
  feature_selection: OutputArtifact[FeatureSelectionArtifact],
  updated_data: OutputArtifact[Examples]):
  """Feature Selection component
    Args (from the module file):
    - NUM_PARAM: Parameter for the corresponding mode in SelectorFunc
        example: value of 'k' in SelectKBest
    - TARGET_FEATURE: Name of the feature containing target data
    - SelectorFunc: Selector function for univariate feature selection
      example: SelectKBest, SelectPercentile from sklearn.feature_selection
    - ScoreFunc: Scoring function for various features with INPUT_DATA and OUTPUT_DATA as parameters
  """


  # importing the required functions and variables from the module file
  modules = importlib.import_module(module_file)
  mod_names = ["NUM_PARAM", "TARGET_FEATURE", "SelectorFunc", "ScoreFunc"]
  NUM_PARAM, TARGET_FEATURE, SelectorFunc, ScoreFunc = [getattr(modules, i) for i in mod_names]

  # uri for the required data
  train_uri = artifact_utils.get_split_uri([orig_examples], 'train')
  np_dataset = get_data_from_TFRecords(train_uri)
  FEATURE_KEYS, TARGET_DATA, INPUT_DATA = data_preprocessing(np_dataset, TARGET_FEATURE)

  # Select features based on scores
  selector = SelectorFunc(ScoreFunc, k=NUM_PARAM)
  selector.fit_transform(INPUT_DATA, TARGET_DATA)

  # adding basic info to the updated example artifact as output
  updated_data.split_names = orig_examples.split_names
  updated_data.span = orig_examples.span

  # generate a list of selected features by matching FEATURE_KEYS to selected indices
  selected_features = [val for (idx, val) in enumerate(FEATURE_KEYS) if idx in selector.get_support(indices=True)]

  # convert string to array
  split_arr = eval(orig_examples.split_names)

  # update feature per split
  for split in split_arr:
    split_uri = artifact_utils.get_split_uri([orig_examples], split)
    new_split_uri = artifact_utils.get_split_uri([updated_data], split)
    os.mkdir(new_split_uri)

    for file in get_file_list(split_uri):
      split_dataset = tf.data.TFRecordDataset(os.path.join(split_uri,file), compression_type='GZIP')

    # write the TFRecord
      with tf.io.TFRecordWriter(path = os.path.join( new_split_uri, file), options="GZIP") as writer:
        for split_record in split_dataset.as_numpy_iterator():
          example = tf.train.Example()
          example.ParseFromString(split_record)

          updated_example = update_example(selected_features, example)
          writer.write(updated_example.SerializeToString())

  

  # get scores and p-values for artifacts
  selector_scores = selector.scores_
  selector_p_values = selector.pvalues_

  # merge scores and pvalues with feature keys to create a dictionary
  selector_scores_dict = dict(zip(FEATURE_KEYS, selector_scores))
  selector_pvalues_dict = dict(zip(FEATURE_KEYS, selector_p_values))

  # populate artifact with the required properties
  feature_selection.scores = selector_scores_dict
  feature_selection.p_values = selector_pvalues_dict
  feature_selection.selected_features = selected_features

