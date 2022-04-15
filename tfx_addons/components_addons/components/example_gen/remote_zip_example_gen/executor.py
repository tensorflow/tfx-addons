# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Remote Zip csv based TFX example gen executor."""
import datetime
import os
from typing import Any, Dict, Union
from tfx.dsl.io import fileio
from tfx.components.example_gen.csv_example_gen.executor import _CsvToExample
import pandas as pd
from absl import logging
import apache_beam as beam
import tensorflow as tf
from zipfile import ZipFile
from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.types import standard_component_specs
from tfx_addons.components_addons.types.standard_component_specs import REMOTE_ZIP_FILE_URI_KEY


def download_dataset(zip_file_uri: str, download_dir: str) -> str:
    """Downloads a dataset from a given uri and saves it to a Download directory.

  Args:
    zip_file_uri: The uri of the dataset to download.
    download_dir: The directory to save the dataset to.

  Returns:
    The path to the downloaded dataset.

  Raises:
    ValueError: If the dataset cannot be downloaded.
  """

    try:
        print("Downloading file in ", download_dir)
        # Creating download_dir if not exists
        if os.path.exists(download_dir):
            shutil.rmtree(download_dir)
        os.makedirs(download_dir, exist_ok=True)
        # Obtaining zip file path to download zip file
        zip_file_path = os.path.join(download_dir, os.path.basename(zip_file_uri))
        request.urlretrieve(zip_file_uri, zip_file_path)
        return zip_file_path
    except Exception as e:
        raise ValueError('Failed to download dataset from {}. {}'.format(zip_file_uri, e))


def extract_zip_file(zip_file_path: str, extract_dir: str, zip_file_read_mode: str = "r") -> None:
    """Extracts a zip file to a directory.

  Args:
  zip_file_path: The path to the zip file.
  output_dir: The directory to extract the zip file to.
  """
    try:
        print("Extracting file in ", extract_dir)
        os.makedirs(extract_dir, exist_ok=True)
        with ZipFile(zip_file_path, zip_file_read_mode) as zip_file:
            zip_file.extractall(extract_dir)
    except Exception as e:
        raise e


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _ZipToExample(  # pylint: disable=invalid-name
        pipeline: beam.Pipeline, exec_properties: Dict[str, Any],
        split_pattern: str) -> beam.pvalue.PCollection:
    """Read remote zip csv files and transform to TF examples.

  Note that each input split will be transformed by this function separately.

  Args:
    pipeline: beam pipeline.
    exec_properties: A dict of execution properties.
      - input_base: input dir that contains Avro data.
    split_pattern: Split.pattern in Input config, glob relative file pattern
      that maps to input files with root directory given by input_base.

  Returns:
    PCollection of TF examples.
  """
    # directory to extract zip file

    input_base_uri = os.path.join(exec_properties[standard_component_specs.INPUT_BASE_KEY])

    # remote zip file uri to download zip file
    zip_file_uri = exec_properties[REMOTE_ZIP_FILE_URI_KEY]

    # downloading zip file from zip file uri into input_base_uri location
    zip_file_path = download_dataset(zip_file_uri, input_base_uri)

    # extract zip files at input_base_uri
    extract_zip_file(zip_file_path, input_base_uri)
    os.remove(zip_file_path)
    return _CsvToExample(exec_properties=exec_properties, split_pattern=split_pattern).expand(pipeline=pipeline)


class Executor(BaseExampleGenExecutor):
    """TFX example gen executor for processing remote zip csv format.

  Data type conversion:
    integer types will be converted to tf.train.Feature with tf.train.Int64List.
    float types will be converted to tf.train.Feature with tf.train.FloatList.
    string types will be converted to tf.train.Feature with tf.train.BytesList
      and utf-8 encoding.

    Note that,
      Single value will be converted to a list of that single value.
      Missing value will be converted to empty tf.train.Feature().

    For details, check the dict_to_example function in example_gen.utils.


  Example usage:

    from tfx.components.base import executor_spec
    from tfx.components.example_gen.component import
    FileBasedExampleGen
    from tfx.components.example_gen.custom_executors import
    avro_executor

    example_gen = FileBasedExampleGen(
        input_base=avro_dir_path,
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            avro_executor.Executor))
  """

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        """Returns PTransform for avro to TF examples."""
        return _ZipToExample
