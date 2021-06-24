import os
import tempfile
import executor
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tfx
import filecmp
from absl.testing import absltest
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils

INPUT_KEY = 'input_data'
OUTPUT_KEY = 'output_data'
LABEL_KEY = 'label'
NAME_KEY = 'name'
SPLIT_KEY = 'splits'
COPY_KEY = 'copy_others'
SHARDS_KEY = 'shards'

class ExecutorTest(absltest.TestCase):
  def _validate_output(self, output):
    pass

  def _validate_same(self, dir0, dir1):
    comp = filecmp.dircmp(dir0, dir1)
    self.assertTrue(comp.left_only == [])
    self.assertTrue(comp.right_only == [])
    self.assertTrue(comp.diff_files == [])
  
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
      LABEL_KEY: 'company',
      NAME_KEY: 'undersampling',
      SPLIT_KEY: json_utils.dumps(['train']), # List needs to be serialized before being passed into Do function.
      COPY_KEY: True,
      SHARDS_KEY: 1,
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

    # Check outputs.
    self._validate_output(os.path.join(output.uri, 'Split-train'))
    self._validate_same(os.path.join(output.uri, 'Split-eval'), artifact_utils.get_split_uri([examples], 'eval'))

    self.assertTrue(
        fileio.exists(os.path.join(output.uri, 'Split-train')))

    self.assertTrue(
        fileio.exists(os.path.join(output.uri, 'Split-eval')))


if __name__ == '__main__':
  tf.test.main()
