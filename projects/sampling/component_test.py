import tensorflow as tf
import tensorflow_data_validation as tfdv
import component
from absl.testing import absltest
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts

INPUT_KEY = 'input_data'
OUTPUT_KEY = 'output_data'
LABEL_KEY = 'label'
NAME_KEY = 'name'
SPLIT_KEY = 'splits'
COPY_KEY = 'copy_others'
SHARDS_KEY = 'shards'

class ComponentTest(absltest.TestCase):
  def testConstruct(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    
    under = component.UndersamplingComponent(
        input_data=channel_utils.as_channel([examples]),
        label='label')

    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME, under.outputs[OUTPUT_KEY].type_name)
    self.assertEqual(
        under.spec.exec_properties[SPLIT_KEY], ['train'])
    self.assertEqual(
        under.spec.exec_properties[LABEL_KEY], 'label')

  def testConstructWithOptions(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    under = component.UndersamplingComponent(
        input_data=channel_utils.as_channel([examples]),
        label='test_label',
        name='test_name',
        splits=['train', 'eval'],
        copy_others=False,
        shards=10)

    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME, under.outputs[OUTPUT_KEY].type_name)
    self.assertEqual(
        under.spec.exec_properties[LABEL_KEY], 'test_label')
    self.assertEqual(
        under.spec.exec_properties[NAME_KEY], 'test_name')
    self.assertEqual(
        under.spec.exec_properties[SPLIT_KEY], ['train', 'eval'])
    self.assertEqual(
        under.spec.exec_properties[COPY_KEY], False)
    self.assertEqual(
        under.spec.exec_properties[SHARDS_KEY], 10)

if __name__ == '__main__':
  tf.test.main()
