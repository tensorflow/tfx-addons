# Copyright 2023 Google LLC. All Rights Reserved.
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
"""CopyExampleGen custom component.

This component will accept tfrecord files and register them as an
Examples Artifact for downstream components to use. CopyExampleGen accepts
a dictionary where keys are the split-names and their respective value is a
uri to the folder that contains the tfrecords file(s).

User will need to create a dictionary of type Dict[str, str], in this case
we will title this dictionary 'tfrecords_dict' and assign it to a dictionary:

  tfrecords_dict: Dict[str, str]={
      "train":"gs://path/to/examples/Split-train/",
      "eval":"gs://path/to/examples/Split-eval/"
    }

Currently tfx.dsl.components.Parameter only supports primitive types therefore,
in order to properly use CopyExampleGen, the 'input_dict' of type Dict[str, str]
needs to be converted into a JSON str. We can do this by simply using
'json.dumps()' by adding 'tfrecords_dict' in as a parameter like so:

  copy_example=component.CopyExampleGen(
      input_json_str=json.dumps(tfrecords_dict)
    )

"""
import json
import os
from typing import List

from tfx import v1 as tfx
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.io import fileio
from tfx.v1.types.standard_artifacts import Examples


def _split_names_property_builder(split_names_list: List):
  """
  _split_names_property_builder() creates a unique string of split-names for
  input to output_example.split_names property like so: '["train","eval"]'

  This format is unique and required for ExampleArtifact property 'split_names'

  """

  property_split_names = "["
  property_split_names_len = len(split_names_list) - 1
  index = 0

  for name in split_names_list:
    if index == property_split_names_len:
      property_split_names = (f'{property_split_names}"{name}"]')
      break
    property_split_names = (f'{property_split_names}"{name}",')
    index += 1
  return property_split_names


@component
def CopyExampleGen(  # pylint: disable=C0103
    input_json_str: tfx.dsl.components.Parameter[str],
    output_example: tfx.dsl.components.OutputArtifact[Examples]
) -> tfx.dsl.components.OutputDict():
  """
  CopyExampleGen first converts the string input to a type Dict and extracts
  the keys from the dictionary, input_dict, and creates a string containing
  the names. This string is assigned to the output_example.split_uri property
  to register split_names property.

  This component then creates a directory folder for each name in split_name.
  Following the creation of the `Split-name` folder, the files in the uri path
  will then be copied into the designated `Split-name` folder.

  """

  # Convert primitive type str to Dict[str, str]
  input_dict = json.loads(input_json_str)

  # Creates directories from the split-names and tfrecord uris provided into
  # output_example.split_names property
  split_names = []
  tfrecords_list = []

  for split_label, split_tfrecords_uri in input_dict.items():
    split_names.append(split_label)

    # Build split_names in required Examples Artifact properties format
    artifact_property_split_names = _split_names_property_builder(split_names)
    output_example.split_names = str(artifact_property_split_names)

    # Create Split-name folder name and create directory
    output_example_uri = output_example.uri
    split_value = (f"/Split-{split_label}/")
    fileio.mkdir(f"{output_example_uri}{split_value}")

    # Pull all files from uri
    tfrecords_list = fileio.glob(f"{split_tfrecords_uri}*.gz")

    # Copy files into folder directories
    for tfrecord in tfrecords_list:
      file_name = os.path.basename(os.path.normpath(tfrecord))
      file_destination = (f"{output_example_uri}{split_value}{file_name}")
      fileio.copy(tfrecord, file_destination, True)
      