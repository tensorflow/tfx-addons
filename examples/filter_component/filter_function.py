"""Filters the data from input data by using the filter function.

Args:
  x_list: Input list of data to be filtered.


Returns:
  filtered list

"""
# TODO: This license is not consistent with the license used in the project.
#       Delete the inconsistent license and above line and rerun pre-commit to insert a good license.
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

def filter_function(x_list):
    """Filters the data from input data by using the filter function.

    Args:
      x_list: Input list of data to be filtered.


    Returns:
      filtered list

    """
    new_list = []
    for element in x_list:
        if element['label'] == [0]:
            new_list.append(element)
    return new_list