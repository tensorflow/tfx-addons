import tensorflow as tf
import component
from absl.testing import absltest

from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils


class ComponentTest(absltest.TestCase):
  def testConstruct(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    params = {
        component.UNDERSAMPLER_INPUT_KEY: channel_utils.as_channel([examples]),
        component.UNDERSAMPLER_LABEL_KEY: 'label'
    }

    under = component.Undersampler(**params)

    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME, under.outputs[component.UNDERSAMPLER_OUTPUT_KEY].type_name)
    self.assertEqual(
        under.spec.exec_properties[component.UNDERSAMPLER_SPLIT_KEY], json_utils.dumps(['train']))
    self.assertEqual(
        under.spec.exec_properties[component.UNDERSAMPLER_LABEL_KEY], 'label')

  def testConstructWithOptions(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    params = {
        component.UNDERSAMPLER_INPUT_KEY: channel_utils.as_channel([examples]),
        component.UNDERSAMPLER_LABEL_KEY: 'test_label',
        component.UNDERSAMPLER_NAME_KEY: 'test_name',
        component.UNDERSAMPLER_SPLIT_KEY: ['train', 'eval'],
        component.UNDERSAMPLER_COPY_KEY: False,
        component.UNDERSAMPLER_SHARDS_KEY: 10,
        component.UNDERSAMPLER_CLASSES_KEY: ['label']
    }

    under = component.Undersampler(**params)

    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME, under.outputs[component.UNDERSAMPLER_OUTPUT_KEY].type_name)
    self.assertEqual(
        under.spec.exec_properties[component.UNDERSAMPLER_LABEL_KEY], 'test_label')
    self.assertEqual(
        under.spec.exec_properties[component.UNDERSAMPLER_NAME_KEY], 'test_name')
    self.assertEqual(
        under.spec.exec_properties[component.UNDERSAMPLER_SPLIT_KEY], json_utils.dumps(['train', 'eval']))
    self.assertEqual(
        under.spec.exec_properties[component.UNDERSAMPLER_COPY_KEY], False)
    self.assertEqual(
        under.spec.exec_properties[component.UNDERSAMPLER_SHARDS_KEY], 10)
    self.assertEqual(
        under.spec.exec_properties[component.UNDERSAMPLER_CLASSES_KEY], json_utils.dumps(['label']))


if __name__ == '__main__':
  tf.test.main()
