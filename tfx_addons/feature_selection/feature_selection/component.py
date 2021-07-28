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
from tfx.dsl.component.experimental.decorators import component
from tfx.types import artifact, standard_artifacts

from module_file import _FEATURE_KEYS, _NUM_PARAM, _INPUT_DATA, _TARGET_DATA, SelectorFunc, ScoreFunc

# _selection_modes = {
#   'percentile': SelectPercentile,
#   'k_best': SelectKBest,
#   'fpr': SelectFpr,
#   'fdr': SelectFdr,
#   'fwe': SelectFwe
# }

"""Custom Artifact type"""

class FeatureSelectionArtifact(artifact.Artifact):
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
def FeatureSelection(feature_selection: tfx.dsl.components.OutputArtifact[FeatureSelectionArtifact]):
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
  selector = SelectorFunc(ScoreFunc, k=_NUM_PARAM)
  # selected_data = selector.fit_transform(_INPUT_DATA, _TARGET_DATA)

  # get scores and p-values for artifacts
  selector_scores = selector.scores_
  selector_p_values = selector.pvalues_
  print("!!!!!!!", selector_scores)

  # merge scores and pvalues with feature keys to create a dictionary
  selector_scores_dict = dict(zip(_FEATURE_KEYS, selector_scores))
  selector_pvalues_dict = dict(zip(_FEATURE_KEYS, selector_p_values))
  print("!!!!!!!", selector_scores_dict)
  print("!!!!!!!", selector_pvalues_dict)

  feature_selection.set_json_value_custom_property("scores", selector_scores_dict)
  feature_selection.set_json_value_custom_property("pvalues", selector_pvalues_dict)
