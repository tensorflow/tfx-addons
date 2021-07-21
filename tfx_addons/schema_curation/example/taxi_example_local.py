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
"""Chicago taxi example using TFX schema curation custom component.
base code taken from: https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/hello_world/example/taxi_pipeline_hello.py

This example demonstrate the use of schema curation custom component.
user defined function `schema_fn` defined in `module_file.py` is used
to change feature `tips` from required to optional.

"""

import os
import tempfile
import urllib
from typing import Text

import absl
import tfx
from tfx.components import CsvExampleGen, SchemaGen, StatisticsGen
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local import local_dag_runner

from tfx_addons.schema_curation.component import component

# downloading data and setting up required paths
_data_root = tempfile.mkdtemp(prefix='tfx-data')
DATA_PATH = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/chicago_taxi_pipeline/data/simple/data.csv'
_data_filepath = os.path.join(_data_root, "data.csv")
urllib.request.urlretrieve(DATA_PATH, _data_filepath)

_pipeline_name = 'taxi_pipeline'
_tfx_root = tfx.__path__[0]
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     metadata_path: Text) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=data_root)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  # inferes a schema
  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
                         infer_feature_shape=True)

  # modifies infered schema with use of udf `schema_fn` defined in module file
  schema_curation = component.SchemaCuration(
      schema=schema_gen.outputs['schema'],
      module_file=os.path.join('schemacomponent', 'example', 'module_file.py'))

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[example_gen, statistics_gen, schema_gen, schema_curation],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path))


# To run this pipeline from the python CLI:
#   $python taxi_pipeline_hello.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  local_dag_runner.LocalDagRunner().run(
      _create_pipeline(pipeline_name=_pipeline_name,
                       pipeline_root=_pipeline_root,
                       data_root=_data_root,
                       metadata_path=_metadata_path))
