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
"""Data converter library to convert from source data to serialized data to be stored out."""
import abc
from typing import Any, Dict, Optional

import tensorflow as tf
from google.cloud import bigquery


class _Converter(abc.ABC):
  """Takes in data as a dictionary of string -> value and returns the serialized form
    of whatever representation we want.
    """
  @abc.abstractmethod
  def RowToExampleBytes(self, instance: Any) -> bytes:
    """Generate tf.Example bytes from dictionary.

        Args:
            instance (Any): Data row generated from data source

        Returns:
            bytes: Serialized tf.SequenceExample
        """
    pass

  @abc.abstractmethod
  def RowToSequenceExampleBytes(self, instance: Any) -> bytes:
    """Generate tf.SequenceExample bytes from dictionary.

        Args:
            instance (Any): Data row generated from data source

        Returns:
            bytes: Serialized tf.SequenceExample
        """
    pass


# NB(gcasassaez): Fork from tfx.extensions.google_cloud_big_query.utils
# This is mostly to add support for timestamp since its a common
# format used by Feast.
def row_to_example(  # pylint: disable=invalid-name
    field_to_type: Dict[str, str],
    field_name_to_data: Dict[str, Any]) -> tf.train.Example:
  """Convert bigquery result row to tf example.

  Args:
    field_to_type: The name of the field to its type from BigQuery.
    field_name_to_data: The data need to be converted from BigQuery that
      contains field name and data.

  Returns:
    A tf.train.Example that converted from the BigQuery row. Note that BOOLEAN
    type in BigQuery result will be converted to int in tf.train.Example.

  Raises:
    RuntimeError: If the data type is not supported to be converted.
      Only INTEGER, BOOLEAN, FLOAT, STRING is supported now.
  """
  feature = {}
  for key, value in field_name_to_data.items():
    data_type = field_to_type[key]

    if value is None:
      feature[key] = tf.train.Feature()
      continue

    value_list = value if isinstance(value, list) else [value]
    if data_type in ('INTEGER', 'BOOLEAN'):
      feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(
          value=value_list))
    elif data_type == 'FLOAT':
      feature[key] = tf.train.Feature(float_list=tf.train.FloatList(
          value=value_list))
    elif data_type == 'TIMESTAMP':
      feature[key] = tf.train.Feature(float_list=tf.train.FloatList(
          value=[elem.timestamp() for elem in value_list]))
    elif data_type == 'STRING':
      feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[tf.compat.as_bytes(elem) for elem in value_list]))
    else:
      # TODO(jyzhao): support more types.
      raise RuntimeError(
          'BigQuery column type {} is not supported.'.format(data_type))

  return tf.train.Example(features=tf.train.Features(feature=feature))


class _BigQueryConverter(_Converter):
  """Converter class for BigQuery source data"""
  def __init__(self, query: str, project: Optional[str]) -> None:
    client = bigquery.Client(project=project)
    # Dummy query to get the type information for each field.
    query_job = client.query("SELECT * FROM ({}) LIMIT 0".format(query))
    results = query_job.result()
    self._type_map = {}
    for field in results.schema:
      self._type_map[field.name] = field.field_type

  def RowToExampleBytes(self, instance: Dict[str, Any]) -> bytes:
    """Convert bigquery result row to tf example."""
    ex_pb2 = row_to_example(self._type_map, instance)
    return ex_pb2.SerializeToString(deterministic=True)

  def RowToSequenceExampleBytes(self, instance: Dict[str, Any]) -> bytes:
    """Convert bigquery result row to tf sequence example."""
    raise NotImplementedError("SequenceExample not implemented yet.")
