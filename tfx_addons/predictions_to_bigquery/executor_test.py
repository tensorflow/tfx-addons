# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for executor.py."""

import datetime
from typing import Union
from unittest import mock

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from absl.testing import absltest, parameterized
from ml_metadata.proto import metadata_store_pb2
from tensorflow_serving.apis import model_pb2, predict_pb2, prediction_log_pb2
from tfx import types

from tfx_addons.predictions_to_bigquery import executor, utils

logging.set_verbosity(logging.WARNING)

_TIMESTAMP = datetime.datetime.now()


def _create_tf_example(
    features: dict[str, Union[bytes, float, int]]) -> tf.train.Example:
  tf_features = {}
  for key, value in features.items():
    if isinstance(value, bytes):
      tf_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[value]))
    elif isinstance(value, float):
      tf_feature = tf.train.Feature(float_list=tf.train.FloatList(
          value=[value]))
    elif isinstance(value, int):
      tf_feature = tf.train.Feature(int64_list=tf.train.Int64List(
          value=[value]))
    else:
      raise ValueError(f'Unsupported feature type for key {key}:'
                       f' {type(value)} .')
    tf_features[key] = tf_feature
  return tf.train.Example(features=tf.train.Features(feature=tf_features))


def _create_model_spec() -> model_pb2.ModelSpec:
  return model_pb2.ModelSpec(signature_name='serving_default')


def _create_predict_request(
    features: dict[str, Union[bytes, float,
                              int]]) -> predict_pb2.PredictRequest:
  tf_example = _create_tf_example(features)
  request_tensor_proto = tf.make_tensor_proto(
      values=tf_example.SerializeToString(), dtype=tf.string, shape=(1, ))
  return predict_pb2.PredictRequest(
      model_spec=_create_model_spec(),
      inputs={
          'examples': request_tensor_proto,
      },
  )


def _create_predict_response(
    values: list[float]) -> predict_pb2.PredictResponse:
  response_tensor_proto = tf.make_tensor_proto(values=values,
                                               dtype=tf.float32,
                                               shape=(1, len(values)))
  return predict_pb2.PredictResponse(model_spec=_create_model_spec(),
                                     outputs={
                                         'outputs': response_tensor_proto,
                                     })


def _create_prediction_log(
    request: predict_pb2.PredictRequest,
    response: predict_pb2.PredictResponse) -> prediction_log_pb2.PredictionLog:
  predict_log = prediction_log_pb2.PredictLog(request=request,
                                              response=response)
  return prediction_log_pb2.PredictionLog(predict_log=predict_log)


class FilterPredictionToDictFnTest(absltest.TestCase):
  """Tests for FilterPredictionToDictFn class."""
  def setUp(self):
    self.labels = ['l1', 'l2', 'l3']
    self.features = {
        'bytes_feature': tf.io.FixedLenFeature([], dtype=tf.string),
        'float_feature': tf.io.FixedLenFeature([], dtype=tf.float32),
    }
    self.timestamp = datetime.datetime.now()
    self.filter_threshold = 0.5

    self.dofn = executor.FilterPredictionToDictFn(
        features=self.features,
        timestamp=self.timestamp,
        filter_threshold=self.filter_threshold,
        labels=self.labels,
    )

  def test_process(self):
    element = _create_prediction_log(
        request=_create_predict_request(features={
            'bytes_feature': b'a',
            'float_feature': 0.5,
        }),
        response=_create_predict_response([0.1, 0.8, 0.1]),
    )
    output = next(self.dofn.process(element))
    expected = {
        'bytes_feature': 'a',
        'float_feature': 0.5,
        'category_label': 'l2',
        'score': 0.8,
        'datetime': mock.ANY,
    }
    self.assertEqual(expected, output)
    self.assertIsInstance(output['datetime'], datetime.datetime)

  def test_process_below_threshold(self):
    element = _create_prediction_log(
        request=_create_predict_request(features={
            'bytes_feature': b'a',
        }),
        response=_create_predict_response([1 / 3, 1 / 3, 1 / 3]),
    )
    with self.assertRaises(StopIteration):
      _ = next(self.dofn.process(element))

  def test_process_no_labels(self):
    features = {
        'bytes_feature': tf.io.FixedLenFeature([], dtype=tf.string),
    }
    dofn = executor.FilterPredictionToDictFn(
        features=features,
        timestamp=self.timestamp,
        filter_threshold=self.filter_threshold,
        labels=None,
    )
    element = _create_prediction_log(
        request=_create_predict_request(features={
            'bytes_feature': b'a',
        }),
        response=_create_predict_response([0.9]),
    )
    expected = {
        'bytes_feature': 'a',
        'score': 0.9,
        'datetime': mock.ANY,
    }
    output = next(dofn.process(element))
    self.assertEqual(expected, output)


def _make_artifact(uri) -> types.Artifact:
  artifact = types.Artifact(metadata_store_pb2.ArtifactType())
  artifact.uri = uri
  return artifact


def _make_artifact_mapping(
    data_dict: dict[str, str]) -> dict[str, list[types.Artifact]]:
  return {k: [_make_artifact(v)] for k, v in data_dict.items()}


class ExecutorTest(absltest.TestCase):
  """Tests for Executor class."""
  def setUp(self):
    super().setUp()
    self.input_dict = _make_artifact_mapping({
        'transform_graph': '/path/to/transform_output',
        'inference_results': '/path/to/BulkInferrer/inference_results',
        'schema': '/path/to/schema',
    })
    self.output_dict = _make_artifact_mapping(
        {'bigquery_export': '/path/to/bigquery_export'})
    self.exec_properties = {
        'bq_table_name': 'table',
        'bq_dataset': 'dataset',
        'gcp_project': 'project',
        'gcs_temp_dir': 'gs://bucket/temp-dir',
        'expiration_time_delta': 1,
        'filter_threshold': 0.5,
        'table_time_suffix': '%Y%m%d',
        'table_partitioning': True,
        'vocab_label_file': 'vocab_file',
    }

    self.enter_context(
        mock.patch.object(executor, '_get_prediction_log_path', autospec=True))
    self.enter_context(
        mock.patch.object(executor,
                          '_get_tft_output',
                          autospec=True,
                          return_value=object()))
    self.enter_context(
        mock.patch.object(utils, 'get_feature_spec', autospec=True))
    self.enter_context(
        mock.patch.object(executor, '_get_labels', autospec=True))
    self.enter_context(
        mock.patch.object(executor, '_check_bq_table_name', autospec=True))
    self.enter_context(
        mock.patch.object(executor, '_add_bq_table_name_suffix',
                          autospec=True))
    self.enter_context(
        mock.patch.object(executor,
                          '_get_additional_bq_parameters',
                          autospec=True))
    self.enter_context(
        mock.patch.object(utils, 'feature_spec_to_bq_schema', autospec=True))

    self.mock_read_from_tfrecord = self.enter_context(
        mock.patch.object(beam.io, 'ReadFromTFRecord', autospec=True))
    self.mock_pardo = self.enter_context(
        mock.patch.object(beam, 'ParDo', autospec=True))
    self.mock_write_to_bigquery = self.enter_context(
        mock.patch.object(beam.io, 'WriteToBigQuery', autospec=True))

    self.enter_context(
        mock.patch.object(types.Artifact,
                          'set_string_custom_property',
                          autospec=True))

    self.executor = executor.Executor()

  def test_Do(self):
    self.executor.Do(self.input_dict, self.output_dict, self.exec_properties)

    self.mock_read_from_tfrecord.assert_called_once()
    self.mock_pardo.assert_called_once()
    self.mock_write_to_bigquery.assert_called_once()


# pylint: disable=protected-access


class ExecutorModuleTest(parameterized.TestCase):
  """Tests for executor module-level functions."""
  def test_get_prediction_log_path(self):
    inference_results = [_make_artifact('inference_results')]
    expected = 'inference_results/*.gz'
    output = executor._get_prediction_log_path(inference_results)
    self.assertEqual(expected, output)

  @parameterized.named_parameters([
      ('no_inference_results', False),
      ('inference_results', True),
  ])
  def test_get_tft_output(self, has_transform_graph):
    if has_transform_graph:
      transform_graph = [_make_artifact('transform_graph')]
      mock_tftransform_output = self.enter_context(
          mock.patch.object(tft, 'TFTransformOutput', autospec=True))

      _ = executor._get_tft_output(transform_graph)

      mock_tftransform_output.assert_called_once()

    else:
      output = executor._get_tft_output(None)
      self.assertIsNone(output)

  def test_get_labels(self):
    mock_tftransform_output = self.enter_context(
        mock.patch.object(tft, 'TFTransformOutput', autospec=True))
    mock_vocabulary_by_name = (
        mock_tftransform_output.return_value.vocabulary_by_name)
    mock_vocabulary_by_name.return_value = [b'a', b'b']
    vocab_file = 'vocab'
    tft_output = tft.TFTransformOutput('uri')

    output = executor._get_labels(tft_output, vocab_file)

    self.assertEqual(['a', 'b'], output)
    mock_vocabulary_by_name.assert_called_once_with(vocab_file)

  @parameterized.named_parameters([
      ('project_dataset_table', 'gcp_project:bq_dataset.bq_table_name', True),
      ('dataset_table', 'bq_dataset.bq_table_name', True),
      ('table_only', 'bq_table_name', False)
  ])
  def test_check_bq_table_name(self, bq_table_name, is_ok):
    if is_ok:
      try:
        executor._check_bq_table_name(bq_table_name)
      except ValueError:
        self.fail('ValueError was raised unexpectedly.')
    else:
      with self.assertRaises(ValueError):
        executor._check_bq_table_name(bq_table_name)

  @parameterized.named_parameters([('no_timestamp', None, None),
                                   ('timestamp_no_format', _TIMESTAMP, None),
                                   ('timestamp_format', _TIMESTAMP, '%Y%m%d')])
  def test_add_bq_table_name_suffix(self, timestamp, timestring_format):
    basename = 'bq_table'

    output = executor._add_bq_table_name_suffix(basename, timestamp,
                                                timestring_format)

    if timestamp is None:
      expected = basename
      self.assertEqual(expected, output)
    elif timestring_format is None:
      expected = (
          f'bq_table_{timestamp.strftime(executor._DEFAULT_TIMESTRING_FORMAT)}'
      )
      self.assertEqual(expected, output)
    else:
      expected = f'bq_table_{timestamp.strftime(timestring_format)}'
      self.assertEqual(expected, output)

  @parameterized.named_parameters([
      ('no_additional', None, None),
      ('expiration_days_only', 1, None),
      ('table_partitioning_only', None, True),
      ('expiration_table_partitioning', 2, True),
  ])
  def test_get_additional_bq_parameters(self, expiration_days,
                                        table_partitioning):
    output = executor._get_additional_bq_parameters(expiration_days,
                                                    table_partitioning)

    if table_partitioning is None:
      self.assertEqual({}, output)
    if expiration_days is None and table_partitioning is not None:
      expected = {'timePartitioning': {'type': 'DAY'}}
      self.assertEqual(expected, output)
    if expiration_days is not None and table_partitioning is not None:
      # TODO(cezequiel): Use freezegun to set the time a specific value
      expected = {
          'timePartitioning': {
              'type': 'DAY',
              'expirationMs': mock.ANY,
          },
      }
      self.assertEqual(expected, output)


if __name__ == '__main__':
  absltest.main()
