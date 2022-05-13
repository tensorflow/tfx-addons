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
"""Init module for TFX."""

import importlib as _importlib

from .version import _PKG_METADATA, __version__

_ACTIVE_MODULES = [
    "__version__",
] + list(_PKG_METADATA.keys())


def __getattr__(name):  # pylint: disable=C0103
  # PEP-562: Lazy loaded attributes on python modules
  # NB(gcasassaez): We lazy load to avoid issues with dependencies not installed
  # for some subpackes
  if name in _ACTIVE_MODULES:
    return _importlib.import_module("." + name, __name__)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
