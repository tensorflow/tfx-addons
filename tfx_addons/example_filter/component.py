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
from pathy.gcs import BucketClientGCS
import re

# get a list of files for the specified path
def _get_file_list(dir_path,bucket_name='default'):
    if 'gs' == dir_path[:2]:

        client = storage.Client()
        file_list = []
        result = re.search(r"gs://(?P<bucketName>.*?)/(?P<path>.*)",dir_path,re.S|re.U|re.I)
        if not result:
            print('GSC path is not valid')
        for blob in client.list_blobs(result.group('bucketName'), result.group('path')):
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


return_type = TypeVar("return_type")


@component
def FilterComponent(
        input_data: InputArtifact[standard_artifacts.Examples],
        filter_function_str: Parameter[str],
        filtered_data: OutputArtifact[standard_artifacts.Examples],
        output_file: Parameter[str]
) -> OutputDict(list_len=int):
    """Filters the data from input data by using the filter function.

    Args:
      input_data: Input list of data to be filtered.
      output_file: the name of the file to be saved to.
      filter_function_str: Module name of the function that will be used to filter the data.
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
            FilterComponent(input_data ,'my_example.my_filter',output_data)

    Returns:
      len of the list after the filter
           {
             'list_len': len(output_list)
           }

    """
    records = _get_data_from_tfrecords(input_data.uri + "/Split-train")
    filter_function = importlib.import_module(filter_function_str).filter_function
    filtered_data tf.data.Dataset.from_tensor_slices( filter_function(records))
    result_len = len(records)
    filename = 'test.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(filtered_data)

    return {
        'list_len': result_len
    }
