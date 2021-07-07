# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Predict extractor for xgboost models."""

import copy
import os
import pickle
import logging
from typing import Dict, Iterable, List, Text

import apache_beam as beam
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tfx_bsl.tfxio import tensor_adapter
import xgboost as xgb

_PREDICT_EXTRACTOR_STAGE_NAME = 'XGBoostPredict'


def _make_xgboost_predict_extractor(
    eval_shared_model: tfma.EvalSharedModel,
    eval_config: tfma.EvalConfig,
) -> extractor.Extractor:
  """Creates an extractor for performing predictions using a xgboost model.

  The extractor's PTransform loads and runs the serving pickle against
  every extract yielding a copy of the incoming extracts with an additional
  extract added for the predictions keyed by tfma.PREDICTIONS_KEY. The model
  inputs are searched for under tfma.FEATURES_KEY.

  Args:
    eval_shared_model: Shared model (single-model evaluation).

  Returns:
    Extractor for extracting predictions.
  """
  eval_shared_models = model_util.verify_and_update_eval_shared_models(
      eval_shared_model)
  return extractor.Extractor(
      stage_name=_PREDICT_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractPredictions(  # pylint: disable=no-value-for-parameter
          eval_shared_models={m.model_name: m for m in eval_shared_models},
          eval_config=eval_config
      ))


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
class _TFMAPredictionDoFn(model_util.DoFnWithModels):
  """A DoFn that loads the models and predicts."""

  def __init__(self, 
    eval_shared_models: Dict[Text, types.EvalSharedModel], 
    eval_config: tfma.EvalConfig):
    super(_TFMAPredictionDoFn, self).__init__(
        {k: v.model_loader for k, v in eval_shared_models.items()})
    self._eval_config = eval_config

  def setup(self):
    # models are loaded and stored into self._loaded_models below
    super(_TFMAPredictionDoFn, self).setup()
    self._feature_keys = None
    self._label_key = None

    # check to see whether each of these loaded models has a corresponding model spec
    if self._eval_config:
      label_config = self.extract_model_specs()
      for name in self._loaded_models:
        if name not in label_config:
          raise ValueError(f"Missing model spec for loaded model {name}.")

    for name, loaded_model in self._loaded_models.items():
      feature_keys = loaded_model.feature_names
      if self._feature_keys and self._label_key:
        assert self._feature_keys == feature_keys, (
            f'Features mismatch in loaded models. Expected {self._feature_keys}'
            f', got {lfeature_keys} instead.')
        assert self._label_key == label_config[name], (
            f'Label mismatch in loaded models. Expected "{self._label_key}"'
            f', got "{label_config[name]}" instead.')
      elif feature_keys and label_config[name]:
        self._feature_keys = feature_keys
        self._label_key = label_config[name]
      else:
        raise ValueError(f'Missing feature or label keys in loaded model {name}.')

  def extract_model_specs(self):
    label_specs = {}
    for config in self._eval_config.model_specs:
      if config.name:
        label_specs[config.name] = config.label_key
      else: # if the input name to ModelSpec is None, ModelSpec doesn't save it and config.name resolves to ''.
        label_specs[None] = config.label_key
    return label_specs

  def process(self, elem: types.Extracts) -> Iterable[types.Extracts]:
    """Uses loaded models to make predictions on batches of data.

    Args:
      elem: An extract containing batched features.

    Yields:
      Copy of the original extracts with predictions added for each model. If
      there are multiple models, a list of dicts keyed on model names will be
      added, with each value corresponding to a prediction for a single sample.
    """
    # Build feature and label vectors because xgboost cannot read tf.Examples.
    features = []
    labels = []
    result = copy.copy(elem)
    for features_dict in result[constants.FEATURES_KEY]:
      features_row = [features_dict[key] for key in self._feature_keys]
      features.append(np.concatenate(features_row))
      labels.append(features_dict[self._label_key])
    result[constants.LABELS_KEY] = np.concatenate(labels)
    features = xgb.DMatrix(pd.DataFrame(features, columns=self._feature_keys))

    # Generate predictions for each model.
    for model_name, loaded_model in self._loaded_models.items():
      preds = loaded_model.predict(features)
      if len(self._loaded_models) == 1:
        result[constants.PREDICTIONS_KEY] = preds
      elif constants.PREDICTIONS_KEY not in result:
        result[constants.PREDICTIONS_KEY] = [
            {model_name: pred} for pred in preds]
      else:
        for i, pred in enumerate(preds):
          result[constants.PREDICTIONS_KEY][i][model_name] = pred
    yield result


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractPredictions(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_shared_models: Dict[Text, types.EvalSharedModel],
    eval_config: tfma.EvalConfig,
) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions and possibly other tensors to extracts.

  Args:
    extracts: PCollection of extracts with inputs keyed by tfma.INPUTS_KEY.
    eval_shared_models: Shared model parameters keyed by model name.

  Returns:
    PCollection of Extracts updated with the predictions.
  """
  return extracts | 'Predict' >> beam.ParDo(
      _TFMAPredictionDoFn(eval_shared_models, eval_config))


def _custom_model_loader_fn(model_path: Text):
  """Returns a function that loads a xgboost model."""
  def loader(path):
    model = xgb.Booster()
    model.load_model(path)
    return model
  return lambda: loader(model_path)


# TFX Evaluator will call the following functions.
def custom_eval_shared_model(
    eval_saved_model_path, model_name, eval_config,
    **kwargs) -> tfma.EvalSharedModel:
  """Returns a single custom EvalSharedModel."""
  model_path = os.path.join(eval_saved_model_path, 'model.json')
  return tfma.default_eval_shared_model(
      eval_saved_model_path=model_path,
      model_name=model_name,
      eval_config=eval_config,
      custom_model_loader=types.ModelLoader(
          construct_fn=_custom_model_loader_fn(model_path)),
      add_metrics_callbacks=kwargs.get('add_metrics_callbacks'))


def custom_extractors(
    eval_shared_model: tfma.MaybeMultipleEvalSharedModels,
    eval_config: tfma.EvalConfig,
    tensor_adapter_config: tensor_adapter.TensorAdapterConfig,
) -> List[tfma.extractors.Extractor]:
  """Returns default extractors plus a custom prediction extractor."""
  predict_extractor = _make_xgboost_predict_extractor(eval_shared_model, eval_config)
  return tfma.default_extractors(
      eval_shared_model=eval_shared_model,
      eval_config=eval_config,
      tensor_adapter_config=tensor_adapter_config,
      custom_predict_extractor=predict_extractor)


def get_module_file():
    return os.path.abspath(__file__)
