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
"""
Tests for tfx_addons.feast_examplegen.component.
"""

import pytest

try:
  import feast
except ImportError:
  pytest.skip("feast not available, skipping", allow_module_level=True)

from tfx.v1.proto import Input

from tfx_addons.feast_examplegen.component import FeastExampleGen


def test_init_valid():
  entity_query = 'SELECT user FROM fake_db'
  repo_config = feast.RepoConfig(provider='local', project='default')
  FeastExampleGen(repo_config=repo_config,
                  features=['feature1', 'feature2'],
                  entity_query='SELECT user FROM fake_db')
  FeastExampleGen(repo_config=repo_config,
                  features='feature_service1',
                  entity_query='SELECT user FROM fake_db')
  FeastExampleGen(repo_config=repo_config,
                  features=['feature1', 'feature2'],
                  input_config=Input(splits=[
                      Input.Split(name='train', pattern=entity_query),
                      Input.Split(name='eval', pattern=entity_query),
                  ]))


def test_input_and_entity():
  entity_query = 'SELECT user FROM fake_db'
  repo_config = feast.RepoConfig(provider='local', project='default')
  with pytest.raises(RuntimeError):

    FeastExampleGen(repo_config=repo_config,
                    features=['feature1', 'feature2'],
                    entity_query=entity_query,
                    input_config=Input(splits=[
                        Input.Split(name='train', pattern=entity_query),
                        Input.Split(name='eval', pattern=entity_query),
                    ]))
