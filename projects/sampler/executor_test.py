import os
import tempfile
import executor
import tensorflow as tf
import filecmp

import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from absl.testing import absltest

from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils
from tfx.utils import io_utils
from tfx.components.util import tfxio_utils

SAMPLER_INPUT_KEY = 'input_data'
SAMPLER_OUTPUT_KEY = 'output_data'
SAMPLER_LABEL_KEY = 'label'
SAMPLER_NAME_KEY = 'name'
SAMPLER_SPLIT_KEY = 'splits'
SAMPLER_COPY_KEY = 'copy_others'
SAMPLER_SHARDS_KEY = 'shards'
SAMPLER_CLASSES_KEY = 'keep_classes'


class ExecutorTest(absltest.TestCase):
  def _validate_output(self, output, splits, num=1):
    def generate_elements(data):
      for i in range(len(data[list(data.keys())[0]])):
        yield {key: data[key][i][0] if data[key][i] and len(data[key][i]) > 0
          else "" for key in data.keys()}

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

    assert_that(data, equal_to([num]))

  def _validate_same(self, dir0, dir1):
    comp = filecmp.dircmp(dir0, dir1)
    self.assertTrue(comp.left_only == [])
    self.assertTrue(comp.right_only == [])
    self.assertTrue(comp.diff_files == [])

  def _run_exec(self, exec_properties):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', tempfile.mkdtemp()), self._testMethodName)
    fileio.makedirs(output_data_dir)

    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, "example_gen")
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    input_dict = {
        SAMPLER_INPUT_KEY: [examples],
    }

    # Create output dict.
    output = standard_artifacts.Examples()
    output.uri = output_data_dir
    output_dict = {
      SAMPLER_OUTPUT_KEY: [output],
    }

    # Run executor.
    under = executor.Executor()
    under.Do(input_dict, output_dict, exec_properties)

    return output

  def testDo(self):
    exec_properties = {
      SAMPLER_LABEL_KEY: 'company',
      SAMPLER_NAME_KEY: 'undersampling',
      SAMPLER_SPLIT_KEY: json_utils.dumps(['train']), # List needs to be serialized before being passed into Do function.
      SAMPLER_COPY_KEY: True,
      SAMPLER_SHARDS_KEY: 1,
      SAMPLER_CLASSES_KEY: json_utils.dumps([]),
    }

    output = self._run_exec(exec_properties)

    # Check outputs.
    self._validate_output(output, ['train'])
    self.assertTrue(fileio.exists(os.path.join(output.uri, 'Split-train')))
    self.assertTrue(fileio.exists(os.path.join(output.uri, 'Split-eval')))

  def testKeepClasses(self):
    exec_properties = {
      SAMPLER_LABEL_KEY: 'company',
      SAMPLER_NAME_KEY: 'undersampling',
      SAMPLER_SPLIT_KEY: json_utils.dumps(['train']), # List needs to be serialized before being passed into Do function.
      SAMPLER_COPY_KEY: True,
      SAMPLER_SHARDS_KEY: 1,
      SAMPLER_CLASSES_KEY: json_utils.dumps(['None']),
    }

    output = self._run_exec(exec_properties)

    # Check outputs.
    self._validate_output(output, ['train'], 2)

  def testShards(self):
    exec_properties = {
      SAMPLER_LABEL_KEY: 'company',
      SAMPLER_NAME_KEY: 'undersampling',
      SAMPLER_SPLIT_KEY: json_utils.dumps(['train']), # List needs to be serialized before being passed into Do function.
      SAMPLER_COPY_KEY: True,
      SAMPLER_SHARDS_KEY: 20,
      SAMPLER_CLASSES_KEY: json_utils.dumps([]),
    }

    output = self._run_exec(exec_properties)

    # Check outputs.
    out = os.path.join(output.uri, 'Split-train')
    self.assertTrue(len([name for name in os.listdir(out)
      if os.path.isfile(os.path.join(out, name))]) == 20)

  def testSplits(self):
    exec_properties = {
      SAMPLER_LABEL_KEY: 'company',
      SAMPLER_NAME_KEY: 'undersampling',
      SAMPLER_SPLIT_KEY: json_utils.dumps(['train', 'eval']), # List needs to be serialized before being passed into Do function.
      SAMPLER_COPY_KEY: True,
      SAMPLER_SHARDS_KEY: 1,
      SAMPLER_CLASSES_KEY: json_utils.dumps([]),
    }

    output = self._run_exec(exec_properties)

    # Check outputs.
    self._validate_output(output, ['train'])
    self._validate_output(output, ['eval'])

  def testCopy(self):
    exec_properties = {
      SAMPLER_LABEL_KEY: 'company',
      SAMPLER_NAME_KEY: 'undersampling',
      SAMPLER_SPLIT_KEY: json_utils.dumps(['train']), # List needs to be serialized before being passed into Do function.
      SAMPLER_COPY_KEY: False,
      SAMPLER_SHARDS_KEY: 1,
      SAMPLER_CLASSES_KEY: json_utils.dumps([]),
    }

    output = self._run_exec(exec_properties)

    self.assertFalse(fileio.exists(os.path.join(output.uri, 'Split-eval')))

if __name__ == '__main__':
  tf.test.main()
