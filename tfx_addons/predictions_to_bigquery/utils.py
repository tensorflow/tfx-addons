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
"""
Util functions for the Digits Prediction-to-BigQuery component.
"""

import glob
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2


def load_schema(input_path: str) -> Dict:
  """
    Loads a TFX schema from a file and returns schema object.

    Args:
        input_path: Path to the file containing the schema.

    Returns:
        A schema object.
    """

  schema = schema_pb2.Schema()
  schema_text = file_io.read_file_to_string(input_path)
  text_format.Parse(schema_text, schema)
  return tft.tf_metadata.schema_utils.schema_as_feature_spec(
      schema).feature_spec


def _get_compress_type(file_path):
  magic_bytes = {
      b'x\x01': 'ZLIB',
      b'x^': 'ZLIB',
      b'x\x9c': 'ZLIB',
      b'x\xda': 'ZLIB',
      b'\x1f\x8b': 'GZIP'
  }

  two_bytes = open(file_path, 'rb').read(2)
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


def parse_schema(prediction_log_path: str,
                 compression_type: str = 'auto') -> Dict:
  """Parses feature schema from predictions."""

  features = {}

  file_paths = glob.glob(prediction_log_path)
  if compression_type == 'auto':
    compression_type = _get_compress_type(file_paths[0])

  dataset = tf.data.TFRecordDataset(file_paths,
                                    compression_type=compression_type)

  serialized = next(iter(dataset.map(lambda serialized: serialized)))
  seq_ex = tf.train.SequenceExample.FromString(serialized.numpy())

  if seq_ex.feature_lists.feature_list:
    raise NotImplementedError("FeatureLists aren't supported at the moment.")

  for key, feature in seq_ex.context.feature.items():
    features[key] = tf.io.FixedLenFeature((),
                                          _get_feature_type(feature=feature))
  return features


def convert_python_numpy_to_bq_type(python_type: Any) -> str:
  """
    Converts a python type to a BigQuery type.

    Args:
        python_type: A python type.

    Returns:
        A BigQuery type.
    """
  if isinstance(python_type, (int, np.int64)):
    return "INTEGER"
  elif isinstance(python_type, (float, np.float32)):
    return "FLOAT"
  elif isinstance(python_type, (str, bytes)):
    return "STRING"
  elif isinstance(python_type, (bool, np.bool)):
    return "BOOLEAN"
  else:
    raise ValueError("Unsupported type: {python_type}")


def convert_single_value_to_native_py_value(tensor: Any) -> str:
  """
    Converts a Python value to a native Python value.

    Args:
        value: A value.

    Returns:
        Value casted to native Python type.
    """

  if isinstance(tensor, tf.sparse.SparseTensor):
    value = tensor.values.numpy()[0]
    logging.debug(f"sparse value: {value}")
  else:
    value = tensor.numpy()[0]
    logging.debug(f"dense value: {value}")

  if isinstance(value, (int, np.int64, np.int32)):
    return int(value)
  elif isinstance(value, (float, np.float32, np.float64)):
    return float(value)
  elif isinstance(value, str):
    return value
  elif isinstance(value, bytes):
    return value.decode("utf-8")
  elif isinstance(value, (bool, np.bool)):
    return bool(value)
  else:
    raise ValueError(f"Unsupported value type: {value} of type {type(value)}")


def convert_tensorflow_dtype_to_bq_type(tf_dtype: tf.dtypes.DType) -> str:
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


def feature_to_bq_schema(features: Dict[str, Any],
                         required: bool = True) -> List[Dict]:
  """
    Convert a list of features to a list of BigQuery schema fields.

    Args:
        features: A list of features.
        required: Whether the field is required.

    Returns:
        A list of BigQuery schema fields.
    """
  return [{
      "name": feature_name,
      "type": convert_tensorflow_dtype_to_bq_type(feature_def.dtype),
      "mode": "REQUIRED" if required else "NULLABLE",
  } for feature_name, feature_def in features.items()]


def create_annotation_fields(
    label_field_name: str = "category_label",
    score_field_name: str = "score",
    required: bool = True,
    add_datetime_field: bool = True,
) -> List[Dict]:
  """
    Create a list of BigQuery schema fields for the annotation fields.

    Args:
        label_field_name: The name of the label field.
        score_field_name: The name of the score field.
        required: Whether the fields are required.
        add_datetime_field: Whether to add a datetime field.

    Returns:
        A list of BigQuery schema fields.
    """

  label_field = {
      "name": label_field_name,
      "type": "STRING",
      "mode": "REQUIRED" if required else "NULLABLE",
  }

  score_field = {
      "name": score_field_name,
      "type": "FLOAT",
      "mode": "REQUIRED" if required else "NULLABLE",
  }

  fields = [label_field, score_field]

  if add_datetime_field:
    datetime_field = {
        "name": "datetime",
        "type": "TIMESTAMP",
        "mode": "REQUIRED" if required else "NULLABLE",
    }
    fields.append(datetime_field)

  return fields
