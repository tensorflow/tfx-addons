import importlib
import json
import os

import tensorflow as tf
from tfx.dsl.component.experimental.annotations import OutputDict
from tfx.dsl.component.experimental.decorators import component
from tfx.types import artifact, artifact_utils
from tfx.types import standard_artifacts
from tfx.v1.dsl.components import InputArtifact, OutputArtifact, Parameter
from tfx_bsl.coders import example_coder
from typing import Callable, TypeVar


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

# get a list of files for the specified path
def _get_file_list(dir_path):
  file_list = [
      f for f in os.listdir(dir_path)
      if os.path.isfile(os.path.join(dir_path, f))
  ]
  return file_list
# reads and returns data from TFRecords at URI as a list of dictionaries with values as numpy arrays
def _get_data_from_tfrecords(train_uri):
  # get all the data files
  train_uri = [
      os.path.join(train_uri, file_path)
      for file_path in _get_file_list(train_uri)
  ]
  raw_dataset = tf.data.TFRecordDataset(train_uri, compression_type='GZIP')
  print('raw_dataset')
  print(raw_dataset)
  np_dataset = []
  for tfrecord in raw_dataset:
    serialized_example = tfrecord.numpy()
    example = example_coder.ExampleToNumpyDict(serialized_example)
    np_dataset.append(example)

  return np_dataset

# update example with selected features
def _update_example(selected_features, orig_example):
  result = {}
  for key, feature in orig_example.features.feature.items():
    if key in selected_features:
      result[key] = feature

  new_example = tf.train.Example(features=tf.train.Features(feature=result))
  return new_example
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

return_type = TypeVar("return_type")
@component
def MyTrainerComponent(
        training_data: InputArtifact[standard_artifacts.Examples],
        filter_function_str : Parameter[str],
        filtered_data: OutputArtifact[standard_artifacts.Examples],
) -> OutputDict(list_len=int):
  '''My simple trainer component.'''

  records = _get_data_from_tfrecords(training_data.uri+"/Split-train")
  filter_function = importlib.import_module(filter_function_str).filter_function
  records = filter_function(records)
  filtered_data = records
  result_len = 0
  return {
    'list_len': result_len
  }
