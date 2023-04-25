"""
the component for filter addon
"""
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
# TODO: This license is not consistent with the license used in the project.
#       Delete the inconsistent license and above line and rerun pre-commit to insert a good license.
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
import os
import re

import tensorflow as tf
from google.cloud import storage
from tfx.dsl.component.experimental.annotations import OutputDict
from tfx.types import standard_artifacts
from tfx.v1.dsl.components import InputArtifact, Parameter
from tfx_bsl.coders import example_coder


# get a list of files for the specified path
def _get_file_list(dir_path):
  if 'gs' == dir_path[:2]:

    client = storage.Client()
    file_list = []
    result = re.search(r"gs://(?P<bucketName>.*?)/(?P<path>.*)", dir_path,
                       re.S | re.U | re.I)
    if not result:
      print('GSC path is not valid')
    for blob in client.list_blobs(result.group('bucketName'),
                                  result.group('path')):
      file_list.append(str(blob))
  else:
    file_list = [
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]
  return file_list


# reads and returns data from TFRecords at URI as a list of dictionaries with values as numpy arrays
def _get_data_from_tfrecords(train_uri: str):
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


def filter_component(input_data: InputArtifact[standard_artifacts.Examples],
                     filter_function_str: Parameter[str],
                     output_file: Parameter[str]) -> OutputDict(list_len=int):
  """Filters the data from input data by using the filter function.

    Args:
      input_data: Input list of data to be filtered.
      output_file: the name of the file to be saved to.
      filter_function_str: Module name of the function that will be used to
      filter the data.
        Example for the function
            my_example/my_filter.py:

            # filter module must have filter_function implemented
            def filter_function(input_list: Array):
                output_list = []
                for element in input_list:
                    if element.something:
                        output_list.append(element)
                return output_list

            pipeline.py:
            filter_component(input_data ,'my_example.my_filter',output_data)

    Returns:
      len of the list after the filter
           {
             'list_len': len(output_list)
           }

    """
  records = _get_data_from_tfrecords(input_data.uri + "/Split-train")
  filter_function = importlib.import_module(
      filter_function_str).filter_function
  filtered_data = filter_function(records)
  result_len = len(filtered_data)
  new_data = []
  for key in list(filtered_data[0].keys()):
    local_list = []
    for i in range(result_len):
      local_list.append(str(filtered_data[i][key][0]))
    new_data.append(str(local_list))
  writer = tf.data.experimental.TFRecordWriter(output_file)
  writer.write(tf.data.Dataset.from_tensor_slices(new_data).map(lambda x: x))

  return {'list_len': result_len}
