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
"""TFX ExampleValidator component definition."""

from typing import List, Optional, Union
from absl import logging
from tfx import types
from tfx.components.example_validator import executor
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_component_specs
from tfx.utils import json_utils
from tfx.proto import evaluator_pb2
from tfx.orchestration import data_types
import tensorflow_model_analysis as tfma


class ExampleFilter(base_beam_component.BaseBeamComponent):
    """A TFX component to validate input examples.

    The ExampleValidator component uses [Tensorflow Data
    Validation](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv)
    to validate the statistics of some splits on input examples against a schema.

    The ExampleValidator component identifies anomalies in training and serving
    data. The component can be configured to detect different classes of anomalies
    in the data. It can:
      - perform validity checks by comparing data statistics against a schema that
        codifies expectations of the user.

    Schema Based Example Validation
    The ExampleValidator component identifies any anomalies in the example data by
    comparing data statistics computed by the StatisticsGen component against a
    schema. The schema codifies properties which the input data is expected to
    satisfy, and is provided and maintained by the user.

    ## Example
    ```
    # Performs anomaly detection based on statistics and data schema.
    validate_stats = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=infer_schema.outputs['schema'])
    ```

    Component `outputs` contains:
     - `anomalies`: Channel of type `standard_artifacts.ExampleAnomalies`.

    See [the ExampleValidator
    guide](https://www.tensorflow.org/tfx/guide/exampleval) for more details.
    """

    SPEC_CLASS = standard_component_specs.Filter_FunctionSpec
    EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

    def __init__(self,

                 examples: types.BaseChannel,
                 model: Optional[types.BaseChannel] = None,
                 baseline_model: Optional[types.BaseChannel] = None,
                 feature_slicing_spec: Optional[Union[evaluator_pb2.FeatureSlicingSpec,
                                                      data_types.RuntimeParameter]] = None,
                 fairness_indicator_thresholds: Optional[Union[
                     List[float], data_types.RuntimeParameter]] = None,
                 example_splits: Optional[List[str]] = None,
                 eval_config: Optional[tfma.EvalConfig] = None,
                 schema: Optional[types.BaseChannel] = None,
                 module_file: Optional[str] = None,
                 exclude_splits: Optional[List[str]] = None,

                 module_path: Optional[str] = None,
                 filter_functions: () = lambda x: x):
        """Construct an ExampleValidator component.

        Args:
          statistics: A BaseChannel of type `base_component.BaseComponentt`.
          schema: A BaseChannel of type `standard_artifacts.Schema`. _required_
          exclude_splits: Names of splits that the example validator should not
            validate. Default behavior (when exclude_splits is set to None) is
            excluding no splits.
        """
        if exclude_splits is None:
            logging.info('Excluding no splits because exclude_splits is not set.')

        spec = standard_component_specs.Filter_FunctionSpec(
            examples=examples,
            model=model,
            baseline_model=baseline_model,
            feature_slicing_spec=feature_slicing_spec,
            fairness_indicator_thresholds=(
                fairness_indicator_thresholds if isinstance(
                    fairness_indicator_thresholds, data_types.RuntimeParameter) else
                json_utils.dumps(fairness_indicator_thresholds)),
            example_splits=json_utils.dumps(example_splits),
            eval_config=eval_config,
            schema=schema,
            filter_functions=filter_functions,
            module_file=module_file,
            module_path=module_path)
        super().__init__(spec=spec)
