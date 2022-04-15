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
"""TFX RemoteZipCsvExampleGen component definition."""

from typing import Optional, Union

from census_consumer_complaint_custom_component.example_gen.remote_zip_csv_example_gen import executor
from census_consumer_complaint_custom_component.component import RemoteZipFileBasedExampleGen
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2


class RemoteZipCsvExampleGen(RemoteZipFileBasedExampleGen):  # pylint: disable=protected-access
    """Official TFX RemoteZipCsvExampleGen component.

  The remotezipcsv examplegen component takes zip file url of zip compressed csv data, and generates train
  and eval examples for downstream components.

  The remotezipcsv examplegen encodes column values to tf.Example int/float/byte feature.
  For the case when there's missing cells, the csv examplegen uses:
  -- tf.train.Feature(`type`_list=tf.train.`type`List(value=[])), when the
     `type` can be inferred.
  -- tf.train.Feature() when it cannot infer the `type` from the column.

  Note that the type inferring will be per input split. If input isn't a single
  split, users need to ensure the column types align in each pre-splits.

  For example, given the following csv rows of a split:

    header:A,B,C,D
    row1:  1,,x,0.1
    row2:  2,,y,0.2
    row3:  3,,,0.3
    row4:

  The output example will be
    example1: 1(int), empty feature(no type), x(string), 0.1(float)
    example2: 2(int), empty feature(no type), x(string), 0.2(float)
    example3: 3(int), empty feature(no type), empty list(string), 0.3(float)

    Note that the empty feature is `tf.train.Feature()` while empty list string
    feature is `tf.train.Feature(bytes_list=tf.train.BytesList(value=[]))`.

  Component `outputs` contains:
   - `examples`: Channel of type `standard_artifacts.Examples` for output train
                 and eval examples.
  """

    EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

    def __init__(
            self,
            input_base: Optional[str] = None,
            zip_file_uri: Optional[str] = None,
            input_config: Optional[Union[example_gen_pb2.Input,
                                         data_types.RuntimeParameter]] = None,
            output_config: Optional[Union[example_gen_pb2.Output,
                                          data_types.RuntimeParameter]] = None,
            range_config: Optional[Union[range_config_pb2.RangeConfig,
                                         data_types.RuntimeParameter]] = None):
        """Construct a RemoteZipCsvExampleGen component.

    Args:
      input_base: an extract directory containing the CSV files after extraction of downloaded zip file.
      zip_file_uri: Remote Zip file uri to download compressed zip csv file
      input_config: An example_gen_pb2.Input instance, providing input
        configuration. If unset, the files under input_base will be treated as a
        single split.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1.
      range_config: An optional range_config_pb2.RangeConfig instance,
        specifying the range of span values to consider. If unset, driver will
        default to searching for latest span with no restrictions.
    """
        super().__init__(
            input_base=input_base,
            zip_file_uri=zip_file_uri,
            input_config=input_config,
            output_config=output_config,
            range_config=range_config)
