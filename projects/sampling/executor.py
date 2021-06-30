import os
import numpy as np
import tensorflow as tf
import random
import apache_beam as beam
from typing import Any, Dict, List, Text

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.components.util import tfxio_utils
from tfx.dsl.component.experimental.decorators import component
from tensorflow_metadata.proto.v0 import schema_pb2


class UndersamplingExecutor(base_executor.BaseExecutor):
  """Executor for UndersamplingComponent."""

  def Do(self, 
         input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Function that randomly undersamples the 'test' split, and simply
      copies all other splits into the output artifact.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - input_data: A list of type `standard_artifacts.Examples`.
      output_dict: Output dict from key to a list of artifacts, including:
        - output_data: A list of type `standard_artifacts.Examples`.
      exec_properties: A dict of execution properties, including:
        - name: Optional unique name. Necessary iff multiple Hello components
          are declared in the same pipeline.
        - splits: Optional list of splits to undersample. Defaults to ['train'].
    Returns:
      None
    Raises:
      OSError and its subclasses
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    label = exec_properties['label']
    splits = exec_properties['splits']
    copy_others = exec_properties['copy_others']
    shards = exec_properties['shards']
    keep_classes = exec_properties['keep_classes']

    input_artifact = artifact_utils.get_single_instance(
        input_dict['input_data'])
    schema = artifact_utils.get_single_instance(
        input_dict['schema'])
    output_artifact = artifact_utils.get_single_instance(
        output_dict['output_data'])

    if copy_others:
      output_artifact.split_names = input_artifact.split_names
    else:
      output_artifact.split_names = artifact_utils.encode_split_names(splits)

    split_data = {}
    for split in artifact_utils.decode_split_names(input_artifact.split_names):
      uri = artifact_utils.get_split_uri([input_artifact], split)
      split_data[split] = uri
    
    for split, uri in split_data.items():
      if split in splits:        
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        os.mkdir(output_dir)
        self.undersample(split, uri, schema, label, shards, keep_classes, output_dir)
      elif copy_others:
        input_dir = uri
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        os.mkdir(output_dir)
        for filename in fileio.listdir(input_dir):
          input_uri = os.path.join(input_dir, filename)
          output_uri = os.path.join(output_dir, filename)
          io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)

  def parse_schema(self, schema):
    parsed = io_utils.parse_pbtxt_file(os.path.join(schema.uri, "schema.pbtxt"), schema_pb2.Schema())
    return {feat.name: feat.type for feat in parsed.feature}

  def undersample(self, split, uri, schema, label, shards, keep_classes, output_dir):
    def generate_elements(x):
        parsed = tf.train.Example.FromString(x.numpy())
        val = parsed.features.feature['company'].bytes_list.value
        if len(val) > 0:
          label = val[0].decode()
        else:
          label = None
        return (label, parsed)

    def sample(key, value, side=0):
        for item in random.sample(value, side):
            yield item

    def filter_null(item, keep_null=False, null_vals=None):
      keep = not item[0] # Note that 0 is included in this filter!
      if null_vals:
        keep = keep or str(item[0]) in null_vals
      keep ^= not keep_null # null is True if we want to keep nulls
      if keep:
        return item[1]

    files = [os.path.join(uri, name) for name in os.listdir(uri)]
    dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")

    with beam.Pipeline() as p:
        data = (
            p 
            | 'DatasetToPCollection' >> beam.Create(dataset)
            | 'MapToLabel' >> beam.Map(lambda x: generate_elements(x))
        )

        val = (
            data
            | 'CountPerKey' >> beam.combiners.Count.PerKey()
            | 'FilterNull' >> beam.Filter(lambda x: filter_null(x, null_vals=keep_classes))
            | 'Values' >> beam.Values()
            | 'FindMinimum' >> beam.CombineGlobally(lambda elements: min(elements or [-1]))
        )

        res = (
            data 
            | 'GroupBylabel' >> beam.GroupByKey()
            | 'Undersample' >> beam.FlatMapTuple(sample, side=beam.pvalue.AsSingleton(val))
        )
        
        null = (
            data
            | 'ExtractNull' >> beam.Filter(lambda x: filter_null(x, keep_null=True, null_vals=keep_classes))
            | 'NullValues' >> beam.Values()
        )

        merged = ((res, null) | 'Merge PCollections' >> beam.Flatten())

        _ = (
            merged
            | 'Serialize' >> beam.Map(lambda x: x.SerializeToString())
            | 'WriteToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
                os.path.join(output_dir, f'Split-{split}'),
                file_name_suffix='.gz',
                num_shards=shards,
                compression_type=beam.io.filesystem.CompressionTypes.GZIP)
        )
        