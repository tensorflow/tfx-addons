import tensorflow as tf
from absl.testing import absltest

from sampler import spec, component

from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils

class ComponentTest(absltest.TestCase):
  def testConstruct(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    params = {
        spec.SAMPLER_INPUT_KEY: channel_utils.as_channel([examples]),
        spec.SAMPLER_LABEL_KEY: 'label'
    }

    under = component.Sampler(**params)

    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME, under.outputs[spec.SAMPLER_OUTPUT_KEY].type_name)
    self.assertEqual(
        under.spec.exec_properties[spec.SAMPLER_SPLIT_KEY], json_utils.dumps(['train']))
    self.assertEqual(
        under.spec.exec_properties[spec.SAMPLER_LABEL_KEY], 'label')

  def testConstructWithOptions(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    params = {
        spec.SAMPLER_INPUT_KEY: channel_utils.as_channel([examples]),
        spec.SAMPLER_LABEL_KEY: 'test_label',
        spec.SAMPLER_NAME_KEY: 'test_name',
        spec.SAMPLER_SPLIT_KEY: ['train', 'eval'],
        spec.SAMPLER_COPY_KEY: False,
        spec.SAMPLER_SHARDS_KEY: 10,
        spec.SAMPLER_CLASSES_KEY: ['label']
    }

    under = component.Sampler(**params)

    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME, under.outputs[spec.SAMPLER_OUTPUT_KEY].type_name)
    self.assertEqual(
        under.spec.exec_properties[spec.SAMPLER_LABEL_KEY], 'test_label')
    self.assertEqual(
        under.spec.exec_properties[spec.SAMPLER_NAME_KEY], 'test_name')
    self.assertEqual(
        under.spec.exec_properties[spec.SAMPLER_SPLIT_KEY], json_utils.dumps(['train', 'eval']))
    self.assertEqual(
        under.spec.exec_properties[spec.SAMPLER_COPY_KEY], False)
    self.assertEqual(
        under.spec.exec_properties[spec.SAMPLER_SHARDS_KEY], 10)
    self.assertEqual(
        under.spec.exec_properties[spec.SAMPLER_CLASSES_KEY], json_utils.dumps(['label']))


if __name__ == '__main__':
  tf.test.main()
