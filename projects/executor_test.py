import os
import tempfile
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tfx
from absl.testing import absltest
from tfx.dsl.io import fileio
from tfx.components.statistics_gen import component
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
import executor
from tfx.utils import json_utils

INPUT_KEY = 'input_data'
OUTPUT_KEY = 'output_data'
LABEL_KEY = 'label'
NAME_KEY = 'name'
SPLIT_KEY = 'splits'
COPY_KEY = 'copy_others'
SHARDS_KEY = 'shards'

class ExecutorTest(absltest.TestCase):
  def _validate_output(self):
    pass
  
  def testDo(self):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', tempfile.mkdtemp()), self._testMethodName)
    fileio.makedirs(output_data_dir)

    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir)
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    input_dict = {
        INPUT_KEY: [examples],
    }
    
    exec_properties = {
      # List needs to be serialized before being passed into Do function.
      LABEL_KEY: 'company',
      NAME_KEY: 'undersampling',
      SPLIT_KEY: json_utils.dumps(['train']),
      COPY_KEY: True,
      SHARDS_KEY: 10
    }
    
    # Create output dict.
    output = standard_artifacts.Examples()
    output.uri = output_data_dir
    output_dict = {
      OUTPUT_KEY: [output],
    }
    
    # Run executor.
    under = executor.UndersamplingExecutor()
    under.Do(input_dict, output_dict, exec_properties)

    # Check statistics_gen outputs.
    self._validate_output(os.path.join(stats.uri, 'Split-train'))
    self._validate_output(os.path.join(stats.uri, 'Split-eval'))

    self.assertTrue(
        fileio.exists(os.path.join(output.uri, 'Split-train')))

    self.assertTrue(
        fileio.exists(os.path.join(output.uri, 'Split-eval')))


if __name__ == '__main__':
  tf.test.main()
