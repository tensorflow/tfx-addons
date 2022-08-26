"""Tests for TFX Firebase Publisher Custom Component."""

import tensorflow as tf
from tfx.types import standard_artifacts

from tfx_addons.firebase_publisher.component import FirebasePublisher


class FirebasePublisherTest(tf.test.TestCase):
  def testConstruct(self):
    firebase_publisher = FirebasePublisher(
        display_name="test_display_name",
        storage_bucket="storage_bucket"
    )

    self.assertEqual(standard_artifacts.PushedModel.TYPE_NAME,
                     firebase_publisher.outputs['pushed_model'].type_name)

if __name__ == '__main__':
  tf.test.main()