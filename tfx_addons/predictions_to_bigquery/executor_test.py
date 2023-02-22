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
import unittest
from typing import Dict, Sequence, Union
from unittest import mock

import tensorflow as tf
from tensorflow_serving.apis import model_pb2, predict_pb2, prediction_log_pb2

from tfx_addons.predictions_to_bigquery import executor


def _create_tf_example(
    features: Dict[str, Union[bytes, float, int]]) -> tf.train.Example:
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
    features: Dict[str, Union[bytes, float,
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
    values: Sequence[float]) -> predict_pb2.PredictResponse:
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


class FilterPredictionToDictFnTest(unittest.TestCase):
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
        labels=self.labels,
        features=self.features,
        timestamp=self.timestamp,
        filter_threshold=self.filter_threshold,
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


if __name__ == '__main__':
  unittest.main()
