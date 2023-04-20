
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for gcp."""

import os
from typing import List, Text, Optional, Dict
from unittest import mock

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.utils import test_case_utils
from tfx.orchestration.kubeflow import kubeflow_dag_runner


def _create_pipeline(
        pipeline_name: Text,
        pipeline_root: Text,
        data_root: Text,
        trainer_module_file: Text,
        evaluator_module_file: Text,
        ai_platform_training_args: Optional[Dict[Text, Text]],
        ai_platform_serving_args: Optional[Dict[Text, Text]],
        beam_pipeline_args: List[Text],
) -> tfx.dsl.Pipeline:
    """Implements the Penguin pipeline with TFX."""
    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])

    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    # TODO(humichael): Handle applying transformation component in Milestone 3.

    # Uses user-provided Python function that trains a model using TF-Learn.
    # Num_steps is not provided during evaluation because the scikit-learn model
    # loads and evaluates the entire test set at once.
    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
        module_file=trainer_module_file,
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        train_args=tfx.proto.TrainArgs(num_steps=2000),
        eval_args=tfx.proto.EvalArgs(),
        custom_config={
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
                ai_platform_training_args,
        })

    # Get the latest blessed model for model validation.
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
        'latest_blessed_model_resolver')

    # Uses TFMA to compute evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='species')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(
                    class_name='Accuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.6}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ])
        ])

    evaluator = tfx.components.Evaluator(
        module_file=evaluator_module_file,
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        custom_config={
            tfx.extensions.google_cloud_ai_platform.experimental.
            PUSHER_SERVING_ARGS_KEY:
                ai_platform_serving_args,
        })

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            trainer,
            model_resolver,
            evaluator,
            pusher,
        ],
        enable_cache=True,
        beam_pipeline_args=beam_pipeline_args,
    )


class ExampleFilterTest(test_case_utils.TfxTest):
    def setUp(self):
        super(ExampleFilterTest, self).setUp()
        self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

        self._penguin_root = os.path.dirname(__file__)

        self._pipeline_name = 'sklearn_test'
        self._data_root = os.path.join(self._penguin_root, 'data')
        self._trainer_module_file = os.path.join(self._penguin_root,
                                                 'penguin_utils_sklearn.py')
        self._evaluator_module_file = os.path.join(self._penguin_root,
                                                   'sklearn_predict_extractor.py')
        self._pipeline_root = os.path.join(self.tmp_dir, 'tfx', 'pipelines',
                                           self._pipeline_name)
        self._ai_platform_training_args = {
            'project': 'project_id',
            'region': 'us-central1',
        }
        self._ai_platform_serving_args = {
            'model_name': 'model_name',
            'project_id': 'project_id',
            'regions': ['us-central1'],
        }

    @mock.patch('tfx.components.util.udf_utils.UserModuleFilePipDependency.'
                'resolve')
    def testPipelineConstruction(self, resolve_mock):
        # Avoid actually performing user module packaging because relative path is
        # not valid with respect to temporary directory.
        resolve_mock.side_effect = lambda pipeline_root: None

        logical_pipeline = _create_pipeline(  # pylint:disable=protected-access
            pipeline_name=self._pipeline_name,
            pipeline_root=self._pipeline_root,
            data_root=self._data_root,
            trainer_module_file=self._trainer_module_file,
            evaluator_module_file=self._evaluator_module_file,
            ai_platform_training_args=self._ai_platform_training_args,
            ai_platform_serving_args=self._ai_platform_serving_args,
            beam_pipeline_args=[])


        kubeflow_dag_runner.KubeflowDagRunner().run(logical_pipeline)
        file_path = os.path.join(self.tmp_dir, 'sklearn_test.tar.gz')
        self.assertTrue(tfx.dsl.io.fileio.exists(file_path))


if __name__ == '__main__':
    tf.test.main()
