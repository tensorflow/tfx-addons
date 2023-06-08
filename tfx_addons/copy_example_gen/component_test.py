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
"""
Tests for tfx_addons.copy_example_gen.component.
"""
import pytest

from tfx.v1.types.standard_artifacts import Examples
from tfx_addons.copy_example_gen import component

class TestCopyExampleGen:

  def test_empty_input(self) -> None:
    empty_input_json_str = ""
    component.CopyExampleGen(
      input_json_str=empty_input_json_str)

  # TODO(zyang7): Tests to write:
  # 1. the input is even a Dict to begin with
  # 2. then we check if the Dict has content in it
  # 3. check if content has a [str:str] key value pairing
  # 4. check if the value is an accessible gs:// bucket
  # 5. check if there is anything in those buckets
  # 6. check if those buckets are of type .gz