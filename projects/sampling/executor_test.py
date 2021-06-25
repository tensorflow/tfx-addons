import os
import tempfile
import executor
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tfx
import filecmp
import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from absl.testing import absltest
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils
from tfx.utils import io_utils
from tfx.components.util import tfxio_utils

INPUT_KEY = 'input_data'
SCHEMA_KEY = 'schema'
OUTPUT_KEY = 'output_data'
LABEL_KEY = 'label'
NAME_KEY = 'name'
SPLIT_KEY = 'splits'
COPY_KEY = 'copy_others'
SHARDS_KEY = 'shards'

class ExecutorTest(absltest.TestCase):
  def _validate_output(self, output, splits):
    def generate_elements(data):
      for i in range(len(data[list(data.keys())[0]])):
        yield {key: data[key][i][0] if data[key][i] and len(data[key][i]) > 0 else "" for key in data.keys()}

    tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(examples=[output], telemetry_descriptors=[])
    for split in splits:
      tfxio = tfxio_factory(io_utils.all_files_pattern(artifact_utils.get_split_uri([output], split)))

      with beam.Pipeline() as p:
        data = (
          p 
          | 'TFXIORead[%s]' % split >> tfxio.BeamSource()
          | 'DictConversion' >> beam.Map(lambda x: x.to_pydict())
          | 'ConversionCleanup' >> beam.FlatMap(generate_elements)
          | 'MapToLabel' >> beam.Map(lambda x: (x['company'], x)) # change
          | 'CountPerKey' >> beam.combiners.Count.PerKey()
          | 'FilterNull' >> beam.Filter(lambda x: x[0])
          | 'Values' >> beam.Values()
          | 'Distinct' >> beam.Distinct()
          | 'Count' >> beam.combiners.Count.Globally()
        )

    assert_that(data, equal_to([1]))

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
    examples.uri = os.path.join(source_data_dir, "example_gen")
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(source_data_dir, 'schema_gen')

    input_dict = {
        INPUT_KEY: [examples],
        SCHEMA_KEY: [schema],
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
    self._validate_output(output, ['train'])
    self._validate_same(os.path.join(output.uri, 'Split-eval'), artifact_utils.get_split_uri([examples], 'eval'))

    self.assertTrue(
        fileio.exists(os.path.join(output.uri, 'Split-train')))

    self.assertTrue(
        fileio.exists(os.path.join(output.uri, 'Split-eval')))


if __name__ == '__main__':
  tf.test.main()
