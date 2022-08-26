"""Tests for tfx_addons.firebase_publisher.runner."""

from unittest import mock
from unittest.mock import Mock, MagicMock

import tensorflow as tf
from tfx_addons.firebase_publisher import runner

class RunnerTest(tf.test.TestCase):
	def setUp(self):
		super().setUp()

	def testModelExistancy(self):
		model_list = Mock()
		model_list.models = ['model1']
		self.assertTrue(runner._model_exist(model_list))

		model_list.models = []
		self.assertFalse(runner._model_exist(model_list))

	@mock.patch('tfx_addons.firebase_publisher.runner.glob.glob')
	def testModelPathAndType(self, mock_glob):
		tmp_model_path = "/tmp/saved_model"

		mock_glob.return_value = [f"{tmp_model_path}/model.tflite"]
		is_tflite, model_path = runner._get_model_path_and_type(tmp_model_path)
		self.assertTrue(is_tflite)
		self.assertEquals(f"{tmp_model_path}/model.tflite", model_path)

		mock_glob.return_value = []
		is_tflite, model_path = runner._get_model_path_and_type(tmp_model_path)
		self.assertFalse(is_tflite)
		self.assertEquals(tmp_model_path, model_path)

	
	@mock.patch('tfx_addons.firebase_publisher.runner.fileio')
	@mock.patch('tfx_addons.firebase_publisher.runner.tf.io.gfile.GFile')
	def testCheckModelSize(self, mock_gfile, mock_fileio):
		mock_source = Mock()
		mock_source.as_dict.get.return_value = "mock_return"

		mock_gfile().__enter__.return_value.size.return_value = 83886080
		mock_gfile().__exit__ = Mock(return_value=False)

		try:
			runner._check_model_size(mock_source)
		except RuntimeError:
			self.fail("Runtime error occured unexpectedly")

		mock_fileio.remove()
		mock_gfile().__enter__.return_value.size.return_value = 83886081
		mock_gfile().__exit__ = Mock(return_value=False)		
		with self.assertRaises(RuntimeError):
			runner._check_model_size(mock_source)

if __name__ == "__main__":
	tf.test.main()