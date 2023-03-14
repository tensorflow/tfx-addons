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
# This code was originally written by Hannes Hapke (Digits Financial Inc.)
# on Feb. 6, 2023.
"""
Tests around Digits Prediction-to-BigQuery component.
"""

import tensorflow as tf
from tfx.types import channel_utils, standard_artifacts

from . import component


class ComponentTest(tf.test.TestCase):
  def setUp(self):
    super(ComponentTest, self).setUp()
    self._transform_graph = channel_utils.as_channel(
        [standard_artifacts.TransformGraph()])
    self._inference_results = channel_utils.as_channel(
        [standard_artifacts.InferenceResult()])
    self._schema = channel_utils.as_channel([standard_artifacts.Schema()])

  def testConstruct(self):
    # not a real test, just checking if if the component can be
    # instantiated
    _ = component.AnnotateUnlabeledCategoryDataComponent(
        transform_graph=self._transform_graph,
        inference_results=self._inference_results,
        schema=self._schema,
        bq_table_name="gcp_project:bq_database.table",
        vocab_label_file="vocab_txt",
        filter_threshold=0.1,
        table_suffix="%Y",
        table_partitioning=False,
    )


if __name__ == "__main__":
  tf.test.main()
