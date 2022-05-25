# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""
Tests for feast_component.executor.
"""

from __future__ import absolute_import, division, print_function

import datetime
from typing import Optional
from unittest import mock

import apache_beam as beam
import pytest
from apache_beam.testing import util
from google.cloud import bigquery
from google.protobuf.struct_pb2 import Struct

try:
  import feast
  from feast.infra.offline_stores.bigquery import (BigQueryOfflineStoreConfig,
                                                   BigQueryRetrievalJob)

except ImportError:
  pytest.skip("feast not available, skipping", allow_module_level=True)

import tensorflow as tf
from tfx.extensions.google_cloud_big_query import utils
from tfx.proto import example_gen_pb2
from tfx.utils import proto_utils

from tfx_addons.feast_examplegen import executor


class FooRetrievalJob(BigQueryRetrievalJob):
  def to_bigquery(
      self,
      job_config: bigquery.QueryJobConfig = None,  # pylint: disable=unused-argument
      timeout: int = 1800,  # pylint: disable=unused-argument
      retry_cadence: int = 10,  # pylint: disable=unused-argument
  ) -> Optional[str]:  # pylint: disable=invalid-name, unused-argument
    return _MockReadFromFeast()


@beam.ptransform_fn
def _MockReadFromFeast(pipeline, query):  # pylint: disable=invalid-name
  del query  # Unused arg
  mock_query_results = [{
      'timestamp': datetime.datetime.utcfromtimestamp(4.2e6),
      'i': 1,
      'i2': [2, 3],
      'b': True,
      'f': 2.0,
      'f2': [2.7, 3.8],
      's': 'abc',
      's2': ['abc', 'def']
  }]
  return pipeline | beam.Create(mock_query_results)


def _mock_custom_config():
  _REPO_CONFIG_KEY = "repo_conf"  # pylint: disable=invalid-name
  _FEATURE_KEY = "feature_refs"  # pylint: disable=invalid-name

  offline_store = BigQueryOfflineStoreConfig()
  repo_config = feast.RepoConfig(provider='gcp',
                                 project='default',
                                 offline_store=offline_store)
  repo_conf = repo_config.json(exclude={"repo_path"}, exclude_unset=True)
  feature_refs = ['feature1', 'feature2']
  config_struct = Struct()
  config_struct.update({
      _REPO_CONFIG_KEY: repo_conf,
      _FEATURE_KEY: feature_refs
  })
  custom_config_pbs2 = example_gen_pb2.CustomConfig()
  custom_config_pbs2.custom_config.Pack(config_struct)
  custom_config = proto_utils.proto_to_json(custom_config_pbs2)
  return custom_config


def _mock_load_custom_config(custom_config):  # pylint: disable=unused-argument
  offline_store = BigQueryOfflineStoreConfig()
  repo_config = feast.RepoConfig(provider='gcp',
                                 project='default',
                                 offline_store=offline_store)
  repo_conf = repo_config.json(exclude={"repo_path"}, exclude_unset=True)
  feature_refs = ['feature1', 'feature2']

  return {
      executor._REPO_CONFIG_KEY: repo_conf,  # pylint: disable=protected-access
      executor._FEATURE_KEY: feature_refs  # pylint: disable=protected-access
  }


def _mock_get_retrieval_job(entity_query, custom_config):  # pylint: disable=invalid-name, unused-argument
  del entity_query, custom_config  # pylint: disable=unused-argument

  return FooRetrievalJob("SELECT * FROM  `test.test`", "Client",
                         bigquery.QueryJobConfig(dry_run=True), "names")


def _mock_get_historical_features(exec_properties):  # pylint: disable=invalid-name, unused-argument
  del exec_properties  # Unused arg
  return BigQueryRetrievalJob("SELECT * FROM `test.test`", "Client", None,
                              "names")


class ExecutorTest(tf.test.TestCase):
  def setUp(self):
    # Mock BigQuery result schema.
    self._schema = [
        bigquery.SchemaField("timestamp", 'TIMESTAMP', mode='REQUIRED'),
        bigquery.SchemaField('i', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('i2', 'INTEGER', mode='REPEATED'),
        bigquery.SchemaField('b', 'BOOLEAN', mode='REQUIRED'),
        bigquery.SchemaField('f', 'FLOAT', mode='REQUIRED'),
        bigquery.SchemaField('f2', 'FLOAT', mode='REPEATED'),
        bigquery.SchemaField('s', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('s2', 'STRING', mode='REPEATED'),
    ]
    super().setUp()

  def testLoadCustomConfig(self):
    offline_store = BigQueryOfflineStoreConfig()
    repo_config = feast.RepoConfig(provider='gcp',
                                   project='default',
                                   offline_store=offline_store)
    repo_conf = repo_config.json(exclude={"repo_path"}, exclude_unset=True)
    feature_refs = ['feature1', 'feature2']
    config_struct = Struct()
    config_struct.update({
        executor._REPO_CONFIG_KEY: repo_conf,  # pylint: disable=protected-access
        executor._FEATURE_KEY: feature_refs  # pylint: disable=protected-access
    })
    custom_config_pbs2 = example_gen_pb2.CustomConfig()
    custom_config_pbs2.custom_config.Pack(config_struct)
    custom_config = proto_utils.proto_to_json(custom_config_pbs2)
    deseralized_conn = executor._load_custom_config(custom_config)  # pylint: disable=protected-access
    truth_config = _mock_load_custom_config(" ")
    self.assertEqual(deseralized_conn, truth_config)

  @mock.patch.multiple(
      executor,
      _load_custom_config=_mock_load_custom_config,
  )
  def testLoadFeastFeatureStore(self):
    custom_config = executor._load_custom_config(" ")  # pylint: disable=protected-access
    feature_store = executor._load_feast_feature_store(custom_config)  # pylint: disable=protected-access
    self.assertEqual(feature_store.config.project, 'default')  # pylint: disable=protected-access

  @mock.patch.multiple(executor,
                       _load_custom_config=_mock_load_custom_config,
                       _get_retrieval_job=_mock_get_retrieval_job)
  @mock.patch.multiple(
      utils,
      ReadFromBigQuery=_MockReadFromFeast,
  )
  @mock.patch.object(bigquery, 'Client')
  def testFeastToExample(self, mock_client):
    mock_client.return_value.query.return_value.result.return_value.schema = \
      self._schema
    with beam.Pipeline() as pipeline:
      examples = (
          pipeline
          | 'ToTFExample' >> executor._FeastToExampleTransform(  # pylint: disable=protected-access
              exec_properties={
                  '_beam_pipeline_args': [],
                  'custom_config': _mock_custom_config()
              },
              split_pattern='SELECT i, f, s FROM `fake`'))
      feature = {}
      feature['timestamp'] = tf.train.Feature(float_list=tf.train.FloatList(
          value=[4200000]))
      feature['i'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
      feature['i2'] = tf.train.Feature(int64_list=tf.train.Int64List(
          value=[2, 3]))
      feature['b'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
      feature['f'] = tf.train.Feature(float_list=tf.train.FloatList(
          value=[2.0]))
      feature['f2'] = tf.train.Feature(float_list=tf.train.FloatList(
          value=[2.7, 3.8]))
      feature['s'] = tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[tf.compat.as_bytes('abc')]))
      feature['s2'] = tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[tf.compat.as_bytes('abc'),
                 tf.compat.as_bytes('def')]))
      example_proto = tf.train.Example(features=tf.train.Features(
          feature=feature)).SerializeToString(deterministic=True)

      util.assert_that(examples, util.equal_to([example_proto]))


if __name__ == '__main__':
  tf.test.main()
