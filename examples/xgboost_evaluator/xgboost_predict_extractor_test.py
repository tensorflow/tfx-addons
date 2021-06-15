# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for the custom xgboost Evaluator module."""

import os
import pickle
import pandas as pd
import xgboost as xgb
import apache_beam as beam
from apache_beam.testing import util
from google.protobuf import text_format
import tensorflow as tf
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.extractors import features_extractor
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util

import xgboost_predict_extractor


class XGBoostPredictExtractorTest(testutil.TensorflowModelAnalysisTest):

  def setUp(self):
    super(XGBoostPredictExtractorTest, self).setUp()
    self._eval_export_dir = os.path.join(self._getTempDir(), 'eval_export')
    self._create_xgboost_model(self._eval_export_dir)
    self._eval_config = config.EvalConfig(model_specs=[config.ModelSpec(name=None, label_key="label")])
    self._eval_shared_model = (
        xgboost_predict_extractor.custom_eval_shared_model(
            eval_saved_model_path=self._eval_export_dir,
            model_name=None,
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
        schema=self._schema,
        raw_record_column_name=constants.ARROW_INPUT_COLUMN)
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
    feature_extractor = features_extractor.FeaturesExtractor(self._eval_config)
    prediction_extractor = (
        xgboost_predict_extractor._make_xgboost_predict_extractor(
            self._eval_shared_model, self._eval_config))
    with beam.Pipeline() as pipeline:
      predict_extracts = (
          pipeline
          | 'Create' >> beam.Create(
              [e.SerializeToString() for e in self._examples])
          | 'BatchExamples' >> self._tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()  # pylint: disable=no-value-for-parameter
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )

      def check_result(actual):
        try:
          for item in actual:
            self.assertEqual(item['labels'].shape, item['predictions'].shape)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(predict_extracts, check_result)

  def testMakeXGBoostPredictExtractorWithMultiModels(self):
    """Tests that predictions are made from extracts for multiple models."""
    eval_config = config.EvalConfig(model_specs=[
        config.ModelSpec(name='model1', label_key="label"),
        config.ModelSpec(name='model2', label_key="label"),
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

    feature_extractor = features_extractor.FeaturesExtractor(self._eval_config)
    prediction_extractor = (
        xgboost_predict_extractor._make_xgboost_predict_extractor(
            eval_shared_model={
                'model1': eval_shared_model_1,
                'model2': eval_shared_model_2,
            }, 
            eval_config=eval_config))
    with beam.Pipeline() as pipeline:
      predict_extracts = (
          pipeline
          | 'Create' >> beam.Create(
              [e.SerializeToString() for e in self._examples])
          | 'BatchExamples' >> self._tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()  # pylint: disable=no-value-for-parameter
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )

      def check_result(actual):
        try:
          for item in actual:
            self.assertEqual(len(item['labels']), len(item['predictions']))
            self.assertIn('model1', item['predictions'][0])
            self.assertIn('model2', item['predictions'][0])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(predict_extracts, check_result)

  def test_custom_eval_shared_model(self):
    """Tests that an EvalSharedModel is created with a custom xgboost loader."""
    model_file = os.path.basename(self._eval_shared_model.model_path)
    self.assertEqual(model_file, 'model.json')
    model = self._eval_shared_model.model_loader.construct_fn()
    self.assertIsInstance(model, xgb.Booster)

  def test_custom_extractors(self):
    """Tests that the xgboost extractor is used when creating extracts."""
    extractors = xgboost_predict_extractor.custom_extractors(
        self._eval_shared_model, self._eval_config, self._tensor_adapter_config)
    self.assertLen(extractors, 6)
    self.assertIn(
        'XGBoostPredict', [extractor.stage_name for extractor in extractors])

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
