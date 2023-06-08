# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

This component will accept TFRecord files and register them as an
Examples Artifact for downstream components to use. CopyExampleGen accepts
a dictionary where keys are the split-names and their respective value is a
URI to the folder that contains the TFRecords file(s).

TFRecord file(s) in URI must resemble same `.gz` file format as the output of
ExampleGen component.

User will need to create a dictionary of type Dict[str, str], in this case
we will title this dictionary 'tfrecords_dict' and assign it to a dictionary:

  tfrecords_dict: Dict[str, str]={
      "train":"gs://path/to/examples/Split-train/",
      "eval":"gs://path/to/examples/Split-eval/"
    }

'tfx.dsl.components.Parameter' only supports primitive types therefore, in
order to properly use CopyExampleGen, the 'input_dict' of type Dict[str, str]
needs to be converted into a JSON str. We can do this by simply using
'json.dumps()' by adding 'tfrecords_dict' in as a parameter like so:

  copy_example=component.CopyExampleGen(
      input_json_str=json.dumps(tfrecords_dict)
    )

"""
import json
import os

from tfx import v1 as tfx
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.io import fileio
from tfx.v1.types.standard_artifacts import Examples


@component
def CopyExampleGen(  # pylint: disable=C0103
    input_json_str: tfx.dsl.components.Parameter[str],
    output_example: tfx.dsl.components.OutputArtifact[Examples]
  ) -> tfx.dsl.components.OutputDict():
  """Copies the TFRecords from input Split directories to reuse.

  CopyExampleGen first converts the string input to a type Dict and extracts
  the keys from the dictionary, input_dict, and creates a string containing
  the names. This string is assigned to the `output_example.split_uri` property
  to register `split_names` property.

  This component then creates a directory folder for each `name` in
  `split_name`. Following the creation of the `Split-{name}` folder, the files
  in the URI path will then be copied into the designated `Split-{name}` folder.

  Args:
    input_json_str: JSON string containing the split labels (key) and URIs
      containing the split label's TFRecords (value). These TFRecords are copied
      to `output_example`.
    output_example: Output Examples object containing the output `uri` and
      `split_names`.

  Returns:
    TODO(@zyang7): Update after return statement is finalized.
  """

  # Convert primitive type str to Dict[str, str].
  input_dict = json.loads(input_json_str)

  tfrecords_list = []
  output_example_uri = output_example.uri

  for split_label, split_tfrecords_uri in input_dict.items():
    # Create Split-name folder name and create directory.
    split_value = f"/Split-{split_label}/"
    fileio.mkdir(f"{output_example_uri}{split_value}")

    # Pull all files from URI.
    tfrecords_list = fileio.glob(f"{split_tfrecords_uri}*.gz")

    # Copy files into folder directories.
    for tfrecord in tfrecords_list:
      file_name = os.path.basename(os.path.normpath(tfrecord))
      file_destination = f"{output_example_uri}{split_value}{file_name}"
      fileio.copy(tfrecord, file_destination, True)

  # Build split_names in required Examples Artifact properties format.
  example_properties_split_names = "[\"{}\"]".format('","'.join(
      input_dict.keys()))
  output_example.split_names = example_properties_split_names

  # TODO(@zyang7): Check if return statement is needed.
