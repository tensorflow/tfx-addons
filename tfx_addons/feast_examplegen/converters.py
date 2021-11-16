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

from google.cloud import bigquery
from tfx.extensions.google_cloud_big_query import utils


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
    ex_pb2 = utils.row_to_example(self._type_map, instance)
    return ex_pb2.SerializeToString()

  def RowToSequenceExampleBytes(self, instance: Dict[str, Any]) -> bytes:
    """Convert bigquery result row to tf sequence example."""
    raise NotImplementedError("SequenceExample not implemented yet.")
