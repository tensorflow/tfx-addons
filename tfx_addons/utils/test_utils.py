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
""" Util functions to assist with the TFX Addons tests """

from typing import Callable, List

from absl import logging
from tfx import v1 as tfx


def get_tfx_version(version: str) -> List[int]:
  """
  Returns the TFX version as integers.
  """
  return [int(x) for x in version.split('.')]


def disable_test(f: Callable,
                 min_version="1.6.0",
                 max_version="2.0.0") -> Callable:

  """ Decorator to disable tests based on TFX version. """

  current_major, current_minor, current_patch = \
    get_tfx_version(tfx.__version__)
  min_major,min_minor,min_patch = get_tfx_version(min_version)
  max_major, max_minor, max_patch = get_tfx_version(max_version)

  min_condition = (
    current_major >= min_major and
    current_minor >= min_minor and
    current_patch >= min_patch)

  if min_condition:
    if current_major < max_major:
      return f
    if current_major == max_major and current_minor < max_minor:
      return f
    if (current_major == max_major and
        current_minor == max_minor and
        current_patch < max_patch):
      return f

  def _decorator():
    logging.warn(f"{f.__name__} has been disabled due to incompatible TFX version.")
  return _decorator
