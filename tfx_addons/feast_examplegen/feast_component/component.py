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

"""
TFX FeastExampleGen Component Definition

QueryBasedExampleGen
https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/component.py

PrestoExampleGen extends QueryBasedExampleGen
https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/presto_component/component.py

"""

from typing import Optional

from tfx.components.example_gen import component
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from tfx.examples.custom_components.presto_example_gen.presto_component import executor
from tfx.proto import example_gen_pb2

from protos import FeastConfig


class FeastExampleGen(component.QueryBasedExampleGen):    

    EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

    def __init__(self,
                 conn_config: FeastConfig.FeastConnConfig,
                 query: Optional[str] = None,
                 input_config: Optional[example_gen_pb2.Input] = None,
                 output_config: Optional[example_gen_pb2.Output] = None):

        packed_custom_config = example_gen_pb2.CustomConfig()
        packed_custom_config.custom_config.Pack(conn_config)

        output_config = output_config or utils.make_default_output_config(
            input_config)

        super().__init__(
            input_config=input_config,
            output_config=output_config,
            custom_config=packed_custom_config)