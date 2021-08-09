# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""XGBoost Evaluator component."""

from tfx import v1 as tfx

from tfx_addons.xgboost_evaluator import xgboost_predict_extractor


class XGBoostEvaluator(tfx.components.Evaluator):
  """A custom Evaluator component made for XGBoost. Keeps everything the same,
  except inputs the custom module file containing the XGBoost Extractor."""
  def __init__(self, **kwargs):
    if 'module_file' in kwargs:
      raise ValueError('XGBoostEvaluator does not accept custom module_file')
    super().__init__(module_file=xgboost_predict_extractor.get_module_file(),
                     **kwargs)
