# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Internal script to perform a minor release"""
import logging
import os
import sys

logging.getLogger().setLevel(logging.INFO)
# Dynamically load root as module so that we can import version
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import tfx_addons as tfxa  # pylint: disable=wrong-import-position

current_version = tfxa.__version__
major, minor, patch = current_version.split(".")

with open(os.path.join(BASE_DIR, "tfx_addons", "version.py")) as f:
  lines = f.readlines()

with open(os.path.join(BASE_DIR, "tfx_addons", "version.py"), "w") as f:
  for l in lines:
    if l.startswith("_VERSION_SUFFIX"):
      f.write('_VERSION_SUFFIX = "rc0"\n')
    else:
      f.write(l)

print(".".join([major, minor]))
