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

import importlib
import tfx.v1 as tfx
from tfx.dsl.component.experimental.decorators import component
from tfx.types import artifact
from tfx.v1.dsl.components import OutputArtifact, Parameter
# TODO: Why does import from tfx.v1.dsl.components not work?


"""Custom Artifact type"""
class FeatureSelectionArtifact(artifact.Artifact):
  """Output artifact containing feature scores from the Feature Selection component"""
  TYPE_NAME = 'Feature Selection'
  PROPERTIES = {
      'scores': artifact.Property(type=artifact.PropertyType.JSON_VALUE),
      'p_values': artifact.Property(type=artifact.PropertyType.JSON_VALUE),
      'selected_features': artifact.Property(type=artifact.PropertyType.JSON_VALUE),
      'selected_data': artifact.Property(type=artifact.PropertyType.JSON_VALUE)
  }


"""
Feature selection component
"""
@component
def FeatureSelection(module_file: Parameter[str],
    feature_selection: OutputArtifact[FeatureSelectionArtifact]):
  """Feature Selection component"""

  # importing the required functions and variables from
  modules = importlib.import_module(module_file)
  mod_names = ["NUM_PARAM", "INPUT_DATA", "TARGET_DATA", "FEATURE_KEYS", "SelectorFunc", "ScoreFunc"]
  NUM_PARAM, INPUT_DATA, TARGET_DATA, FEATURE_KEYS, SelectorFunc, ScoreFunc = [getattr(modules, i) for i in mod_names]

  # Select features based on scores
  selector = SelectorFunc(ScoreFunc, k=NUM_PARAM)
  selected_data = selector.fit_transform(INPUT_DATA, TARGET_DATA).tolist()

  # generate a list of selected features by matching _FEATURE_KEYS to selected indices
  selected_features = [val for (idx, val) in enumerate(FEATURE_KEYS) if idx in selector.get_support(indices=True)]

  # get scores and p-values for artifacts
  selector_scores = selector.scores_
  selector_p_values = selector.pvalues_

  # merge scores and pvalues with feature keys to create a dictionary
  selector_scores_dict = dict(zip(FEATURE_KEYS, selector_scores))
  selector_pvalues_dict = dict(zip(FEATURE_KEYS, selector_p_values))

  # populate artifact with the required properties
  feature_selection.scores = selector_scores_dict
  feature_selection.p_values = selector_pvalues_dict
  feature_selection.selected_features = selected_features
  feature_selection.selected_data = selected_data
