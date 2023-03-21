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
# This code was originally written by Hannes Hapke (Digits Financial Inc.)
# on Feb. 6, 2023.
"""Schema parsing and conversion routines."""

# TODO(cezequiel): Rename file to schema_utils.py

import os
import re
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
import tensorflow_transform as tft
from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tfx.types import Artifact, artifact_utils

FeatureSpec = Dict[str, Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]]
BigQuerySchema = Dict[str, Any]

_SCHEMA_FILE_NAME = "schema.pbtxt"
_REGEX_CHARS_TO_REPLACE = re.compile(r'[^a-zA-Z0-9_]')


def _get_feature_spec_from_schema_file(input_path: str) -> FeatureSpec:
  """Loads a TFX schema from a file and parses it into a TF feature spec.

  Args:
      input_path: Path to the `_SCHEMA_FILE_NAME` file.

  Returns:
      A `FeatureSpec` object.
  """
  schema = schema_pb2.Schema()
  schema_text = file_io.read_file_to_string(input_path)
  text_format.Parse(schema_text, schema)
  return (
      tft.tf_metadata.schema_utils.schema_as_feature_spec(schema).feature_spec)


def _get_compress_type(file_path: str) -> Optional[str]:
  magic_bytes = {
      b'x\x01': 'ZLIB',
      b'x^': 'ZLIB',
      b'x\x9c': 'ZLIB',
      b'x\xda': 'ZLIB',
      b'\x1f\x8b': 'GZIP'
  }

  with tf.io.gfile.GFile(file_path, 'rb') as input_file:
    two_bytes = input_file.read(2)

  return magic_bytes.get(two_bytes)


def _get_feature_type(feature=None, type_=None):

  if type_:
    return {
        int: tf.int64,
        bool: tf.int64,
        float: tf.float32,
        str: tf.string,
        bytes: tf.string,
    }[type_]

  if feature:
    if feature.HasField('int64_list'):
      return tf.int64
    if feature.HasField('float_list'):
      return tf.float32
    if feature.HasField('bytes_list'):
      return tf.string

  return None


def _get_feature_spec_from_prediction_results(
    prediction_log_path: str) -> FeatureSpec:
  """Parses a TensorFlow feature spec from BulkInferrer prediction results.

  Args:
    prediction_log_path: Path containing BulkInferrer prediction results.

  Returns:
    A `FeatureSpec` object.
  """
  filepath = tf.io.gfile.glob(prediction_log_path)[0]
  compression_type = _get_compress_type(filepath)
  dataset = tf.data.TFRecordDataset([filepath],
                                    compression_type=compression_type)

  for bytes_record in dataset.take(1):
    prediction_log = prediction_log_pb2.PredictionLog.FromString(
        bytes_record.numpy())

  example_bytes = (
      prediction_log.predict_log.request.inputs['examples'].string_val[0])
  example = tf.train.Example.FromString(example_bytes)
  features = {}

  for name, feature_proto in example.features.feature.items():
    feature_dtype = _get_feature_type(feature=feature_proto)
    feature = tf.io.VarLenFeature(dtype=feature_dtype)
    features[name] = feature

  return features


def get_feature_spec(
    schema: Optional[List[Artifact]] = None,
    tft_output: Optional[tft.TFTransformOutput] = None,
    prediction_log_path: Optional[str] = None,
) -> Dict[str, Any]:
  """Returns a TensorFlow feature spec representing the input data schema.

  Specify one of `schema`, `tft_output`, `prediction_log_path` as the source
  for the data schema.

  Args:
    schema: Path to a `_SCHEMA_FILENAME` file.
    tft_output: TensorFlow Transform output path.
    prediction_log_path: Path to a TFRecord file containing inference results.
  """
  if schema:  # note: schema can be an empty list
    schema_uri = artifact_utils.get_single_uri(schema)
    schema_file = os.path.join(schema_uri, _SCHEMA_FILE_NAME)
    return _get_feature_spec_from_schema_file(schema_file)

  if tft_output is not None:
    return tft_output.raw_feature_spec()

  if prediction_log_path is None:
    raise ValueError(
        'Specify one of `schema`, `tft_output` or `prediction_log_path`.')

  return _get_feature_spec_from_prediction_results(prediction_log_path)


def _convert_tensorflow_dtype_to_bq_type(tf_dtype: tf.dtypes.DType) -> str:
  """
    Converts a tensorflow dtype to a BigQuery type string.

    Args:
        tf_dtype: A tensorflow dtype.

    Returns:
        A BigQuery type string.
    """
  if tf_dtype in (tf.int64, tf.int64):
    return "INTEGER"
  elif tf_dtype in (tf.float32, tf.float64):
    return "FLOAT"
  elif tf_dtype == tf.string:
    return "STRING"
  elif tf_dtype == tf.bool:
    return "BOOLEAN"
  else:
    raise ValueError(f"Unsupported type: {tf_dtype}")


def get_bq_field_name_from_key(key: str) -> str:
  field_name = _REGEX_CHARS_TO_REPLACE.sub('_', key)
  return re.sub('_+', '_', field_name).strip('_')


def _feature_spec_to_bq_schema_fields(feature_spec: FeatureSpec,
                                      required: bool = True) -> List[Dict]:
  """Convert a TensorFlow feature spec to a list of BigQuery schema fields.

  Args:
    feature_spec: TensorFlow feature spec.
    required: Whether the field is required.

  Returns:
      A list of BigQuery schema fields.
  """
  return [{
      "name": get_bq_field_name_from_key(feature_name),
      "type": _convert_tensorflow_dtype_to_bq_type(feature_def.dtype),
      "mode": "REQUIRED" if required else "NULLABLE",
  } for feature_name, feature_def in feature_spec.items()]


def _create_annotation_fields(
    *,
    required: bool = True,
    add_label_field: bool = False,
    add_datetime_field: bool = True,
) -> List[Dict]:
  """Creates a list of annotation fields in BigQuery schema formatkjjjj.

  Args:
      label_field_name: The name of the label field.
      score_field_name: The name of the score field.
      required: Whether the fields are required.
      add_datetime_field: Whether to add a datetime field.

  Returns:
      A list of BigQuery schema fields.
  """

  fields = []
  if add_label_field:
    label_field = {
        'name': 'category_label',
        'type': 'STRING',
        'mode': 'REQUIRED' if required else 'NULLABLE',
    }
    fields.append(label_field)

  score_field = {
      'name': 'score',
      'type': 'FLOAT',
      'mode': 'REQUIRED' if required else 'NULLABLE',
  }
  fields.append(score_field)

  if add_datetime_field:
    datetime_field = {
        'name': 'datetime',
        'type': 'TIMESTAMP',
        'mode': 'REQUIRED' if required else 'NULLABLE',
    }
    fields.append(datetime_field)

  return fields


def feature_spec_to_bq_schema(feature_spec: FeatureSpec,
                              required: bool = True,
                              **kwargs: int) -> BigQuerySchema:
  """Converts a TensorFlow feature spec into a BigQuery schema.

  Args:
    feature_spec: TensorFlow feature spec.
    required: If True, mark BigQuery fields as required.
    **kwargs: Additional keyword-arguments to pass to
      `_create_annotation_fields`.

  Returns:
    A `BigQuerySchema` object.
  """
  bq_schema_fields = _feature_spec_to_bq_schema_fields(feature_spec,
                                                       required=required)
  bq_schema_fields.extend(
      _create_annotation_fields(required=required, **kwargs))
  return {"fields": bq_schema_fields}
