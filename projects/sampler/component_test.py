import tensorflow as tf
import component
from absl.testing import absltest

from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils

SAMPLER_INPUT_KEY = 'input_data'
SAMPLER_OUTPUT_KEY = 'output_data'
SAMPLER_LABEL_KEY = 'label'
SAMPLER_NAME_KEY = 'name'
SAMPLER_SPLIT_KEY = 'splits'
SAMPLER_COPY_KEY = 'copy_others'
SAMPLER_SHARDS_KEY = 'shards'
SAMPLER_CLASSES_KEY = 'keep_classes'

class ComponentTest(absltest.TestCase):
  def testConstruct(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    params = {
        SAMPLER_INPUT_KEY: channel_utils.as_channel([examples]),
        SAMPLER_LABEL_KEY: 'label'
    }

    under = component.Sampler(**params)

    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME, under.outputs[SAMPLER_OUTPUT_KEY].type_name)
    self.assertEqual(
        under.spec.exec_properties[SAMPLER_SPLIT_KEY], json_utils.dumps(['train']))
    self.assertEqual(
        under.spec.exec_properties[SAMPLER_LABEL_KEY], 'label')

  def testConstructWithOptions(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    params = {
        SAMPLER_INPUT_KEY: channel_utils.as_channel([examples]),
        SAMPLER_LABEL_KEY: 'test_label',
        SAMPLER_NAME_KEY: 'test_name',
        SAMPLER_SPLIT_KEY: ['train', 'eval'],
        SAMPLER_COPY_KEY: False,
        SAMPLER_SHARDS_KEY: 10,
        SAMPLER_CLASSES_KEY: ['label']
    }

    under = component.sampler(**params)

    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME, under.outputs[SAMPLER_OUTPUT_KEY].type_name)
    self.assertEqual(
        under.spec.exec_properties[SAMPLER_LABEL_KEY], 'test_label')
    self.assertEqual(
        under.spec.exec_properties[SAMPLER_NAME_KEY], 'test_name')
    self.assertEqual(
        under.spec.exec_properties[SAMPLER_SPLIT_KEY], json_utils.dumps(['train', 'eval']))
    self.assertEqual(
        under.spec.exec_properties[SAMPLER_COPY_KEY], False)
    self.assertEqual(
        under.spec.exec_properties[SAMPLER_SHARDS_KEY], 10)
    self.assertEqual(
        under.spec.exec_properties[SAMPLER_CLASSES_KEY], json_utils.dumps(['label']))


if __name__ == '__main__':
  tf.test.main()
