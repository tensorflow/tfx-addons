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

from typing import List

import tfx.v1 as tfx
from sklearn.feature_selection import (SelectFdr, SelectFpr, SelectFwe,
                                       SelectKBest, SelectPercentile)
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.components import Parameter, OutputArtifact
from tfx.types import artifact

_selection_modes = {
    'percentile': SelectPercentile,
    'k_best': SelectKBest,
    'fpr': SelectFpr,
    'fdr': SelectFdr,
    'fwe': SelectFwe
}
"""Custom Artifact type"""


class FeatureSelection(artifact.Artifact):
  """Output artifact containing feature scores from the Feature Selection component"""
  TYPE_NAME = 'Feature Scores'
  PROPERTIES = {
      'scores': {},
      'p_values': {}
  }



"""
Feature selection component using sklearn
"""
@component
def FeatureSelection(input_data: List,
                     target_column: List,
                     feature_selection: OutputArtifact[FeatureSelection],
                     selector_func,
                     score_func,
                     column_names=[],
                     num_param: Parameter[int] = 10,
                     ) -> None:
  """Feature Selection component

    Args:
        input_data: Input features in the form of list[list] where each inner list contains one record
        target_column: The target column which is inferred from `input_data`
        selector_func: feature selector type
        score_func: score function for feature selection. Example: chi2 etc.
        num_params: Parameter of the corresponding `selector_func`
        column_names: to generate artifact dictionaries containing scores
    """

  # Select features based on scores
  selector = _selection_modes[selector_func](score_func, k=num_param)
  selected_data = selector.fit_transform(input_data, target_column)

  # get scores and p-values for artifacts
  selector_scores = selector.scores_
  selector_p_values = selector.pvalues_

  # convert scores and p-values to dictionaries with column names as keys for better comprehensibility
  # add the dictionaries to the artifact
  feature_selection.scores = dict(zip(column_names, selector_scores))
  feature_selection.pvalues = dict(zip(column_names, selector_p_values))

  return selected_data
