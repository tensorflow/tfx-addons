# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the custom xgboost Evaluator module."""

import os

import apache_beam as beam
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
import xgboost as xgb
from apache_beam.testing import util
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.types import channel_utils, standard_artifacts
from tfx_bsl.tfxio import tensor_adapter, test_util

from tfx_addons.xgboost_evaluator import component, xgboost_predict_extractor


class XGBoostPredictExtractorTest(
    tfma.test.testutil.TensorflowModelAnalysisTest):
  def setUp(self):
    """Function that sets up a schema, some examples, and some metadata
    for use in the tests."""

    super().setUp()
    self._eval_export_dir = os.path.join(self._getTempDir(), 'eval_export')
    self._create_xgboost_model(self._eval_export_dir)
    self._eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(name='', label_key="label")])
    self._eval_shared_model = (
        xgboost_predict_extractor.custom_eval_shared_model(
            eval_saved_model_path=self._eval_export_dir,
            model_name='',
            eval_config=self._eval_config))
    self._schema = text_format.Parse(
        """
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "language"
          type: FLOAT
        }
        feature {
          name: "label"
          type: INT
        }
        """, schema_pb2.Schema())
    self._tfx_io = test_util.InMemoryTFExampleRecord(
        schema=self._schema, raw_record_column_name=tfma.ARROW_INPUT_COLUMN)
    self._tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=self._tfx_io.ArrowSchema(),
        tensor_representations=self._tfx_io.TensorRepresentations())
    self._examples = [
        self._makeExample(age=3.0, language=1.0, label=1),
        self._makeExample(age=3.0, language=0.0, label=0),
        self._makeExample(age=4.0, language=1.0, label=1),
        self._makeExample(age=5.0, language=0.0, label=0),
    ]

  def testMakeXGBoostPredictExtractor(self):
    """Tests that predictions are made from extracts for a single model."""
    feature_extractor = tfma.extractors.FeaturesExtractor(self._eval_config)
    prediction_extractor = (
        xgboost_predict_extractor.make_xgboost_predict_extractor(
            self._eval_shared_model, self._eval_config))

    def check_result(item):
      try:
        # Regular assert used due to errors with pytest when using self.AssertEqual
        assert item['labels'].shape == item['predictions'].shape

      except AssertionError as err:
        raise util.BeamAssertException(err)

    with beam.Pipeline() as pipeline:
      _ = (
          pipeline
          | 'Create' >> beam.Create(
              [e.SerializeToString() for e in self._examples])
          | 'BatchExamples' >> self._tfx_io.BeamSource()
          | 'InputsToExtracts' >> tfma.BatchedInputsToExtracts()  # pylint: disable=no-value-for-parameter
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
          | beam.Map(check_result))  # the test

  def testMakeXGBoostPredictExtractorWithMultiModels(self):
    """Tests that predictions are made from extracts for multiple models."""
    eval_config = tfma.EvalConfig(model_specs=[
        tfma.ModelSpec(name='model1', label_key="label"),
        tfma.ModelSpec(name='model2', label_key="label"),
    ])
    eval_export_dir_1 = os.path.join(self._eval_export_dir, '1')
    self._create_xgboost_model(eval_export_dir_1)
    eval_shared_model_1 = xgboost_predict_extractor.custom_eval_shared_model(
        eval_saved_model_path=eval_export_dir_1,
        model_name='model1',
        eval_config=eval_config)
    eval_export_dir_2 = os.path.join(self._eval_export_dir, '2')
    self._create_xgboost_model(eval_export_dir_2)
    eval_shared_model_2 = xgboost_predict_extractor.custom_eval_shared_model(
        eval_saved_model_path=eval_export_dir_2,
        model_name='model2',
        eval_config=eval_config)

    feature_extractor = tfma.extractors.FeaturesExtractor(self._eval_config)
    prediction_extractor = (
        xgboost_predict_extractor.make_xgboost_predict_extractor(
            eval_shared_model={
                'model1': eval_shared_model_1,
                'model2': eval_shared_model_2,
            },
            eval_config=eval_config))

    def check_result(item):
      try:
        # Regular assert used due to errors with pytest when using self.AssertEqual
        assert len(item['labels']) == len(item['predictions'])
        assert 'model1' in item['predictions'][0]
        assert 'model2' in item['predictions'][0]

      except AssertionError as err:
        raise util.BeamAssertException(err)

    with beam.Pipeline() as pipeline:
      _ = (
          pipeline
          | 'Create' >> beam.Create(
              [e.SerializeToString() for e in self._examples])
          | 'BatchExamples' >> self._tfx_io.BeamSource()
          | 'InputsToExtracts' >> tfma.BatchedInputsToExtracts()  # pylint: disable=no-value-for-parameter
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
          | beam.Map(check_result))  # the test

  def test_custom_eval_shared_model(self):
    """Tests that an EvalSharedModel is created with a custom xgboost loader."""
    model_file = os.path.basename(self._eval_shared_model.model_path)
    self.assertEqual(model_file, 'model.json')
    model = self._eval_shared_model.model_loader.construct_fn()
    self.assertIsInstance(model, xgb.Booster)

  def test_custom_extractors(self):
    """Tests that the xgboost extractor is used when creating extracts."""
    extractors = xgboost_predict_extractor.custom_extractors(
        self._eval_shared_model, self._eval_config,
        self._tensor_adapter_config)
    self.assertLen(extractors, 6)
    self.assertIn('XGBoostPredict',
                  [extractor.stage_name for extractor in extractors])

  def test_component(self):
    examples = standard_artifacts.Examples()
    model_exports = standard_artifacts.Model()
    evaluator = component.XGBoostEvaluator(
        examples=channel_utils.as_channel([examples]),
        model=channel_utils.as_channel([model_exports]),
        example_splits=['eval'])

    module_file = xgboost_predict_extractor.get_module_file()
    self.assertEqual(standard_artifacts.ModelEvaluation.TYPE_NAME,
                     evaluator.outputs['evaluation'].type_name)
    self.assertEqual(module_file, evaluator.exec_properties["module_file"])

  def _create_xgboost_model(self, eval_export_dir):
    """Creates and pickles a toy xgboost model.

    Args:
        eval_export_dir: Directory to store a pickled xgboost model. This
            directory is created if it does not exist.
    """
    train = pd.DataFrame({"age": [3, 0], "language": [4, 1]})
    label = pd.DataFrame({"label": [0, 1]})
    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    train = xgb.DMatrix(train, label=label)
    model = xgb.train(param, train)

    os.makedirs(eval_export_dir)
    model.save_model(os.path.join(eval_export_dir, "model.json"))


if __name__ == '__main__':
  tf.test.main()
