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
"""Supplement for palmer penguins example with specific feature modification.
This module file will be used in the feature selection component example.
"""
from sklearn.feature_selection import \
    SelectKBest as SelectorFunc  # pylint: disable=W0611
from sklearn.feature_selection import chi2

SELECTOR_PARAMS = {"score_func": chi2, "k": 2}
TARGET_FEATURE = 'species'
