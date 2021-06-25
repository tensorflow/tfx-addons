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
         schema: Dict[Text, List[types.Artifact]],
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

    input_artifact = artifact_utils.get_single_instance(
        input_dict['input_data'])
    output_artifact = artifact_utils.get_single_instance(
        output_dict['output_data'])

    if copy_others:
      output_artifact.split_names = input_artifact.split_names
    else:
      output_artifact.split_names = artifact_utils.encode_split_names(splits)

    split_data = {}
    schema = self.parse_schema(schema)
    tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(examples=[input_artifact], telemetry_descriptors=[])

    for split in artifact_utils.decode_split_names(input_artifact.split_names):
      uri = artifact_utils.get_split_uri([input_artifact], split)
      split_data[split] = [uri, tfxio_factory(io_utils.all_files_pattern(artifact_utils.get_split_uri([input_artifact], split)))]
    
    for split, data in split_data.items():
      if split in splits:        
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        os.mkdir(output_dir)
        self.undersample(split, data[1], schema, label, shards, output_dir)
      elif copy_others:
        input_dir = data[0]
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        os.mkdir(output_dir)
        for filename in fileio.listdir(input_dir):
          input_uri = os.path.join(input_dir, filename)
          output_uri = os.path.join(output_dir, filename)
          io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)

  def parse_schema(self, schema):
    # schema = schema_gen.outputs['schema']
    parsed = io_utils.parse_pbtxt_file(os.path.join(schema._artifacts[0].uri, "schema.pbtxt"), schema_pb2.Schema())
    return {feat.name: feat.type for feat in parsed.feature}

  def undersample(self, split, tfxio, schema, label, shards, output_dir):
    def generate_elements(data):
      for i in range(len(data[list(data.keys())[0]])):
            yield {key: data[key][i][0] if data[key][i] and len(data[key][i]) > 0 else "" for key in data.keys()}

    def sample(key, value, side=0):
        for item in random.sample(value, side):
            yield item
            
    def convert_to_tfexample(data, schema):
        features = dict()
        for key, val in data.items():
            if schema[key] == 2:
                features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data[key]]))
            elif schema[key] == 3:
                features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[data[key]]))
            elif schema[key] == 1:
                # features[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(data[key])]))
                features[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[key]]))
            else:
                features[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[key]]))
        return tf.train.Example(features=tf.train.Features(feature=features))

    with beam.Pipeline() as p:
        data = (
            # TODO: convert to list and back using a schema to save key space?
            p 
            | 'TFXIORead[%s]' % split >> tfxio.BeamSource()
            | 'DictConversion' >> beam.Map(lambda x: x.to_pydict())
            | 'ConversionCleanup' >> beam.FlatMap(generate_elements)
            | 'MapToLabel' >> beam.Map(lambda x: (x[label], x))
        )

        val = (
            data
            | 'CountPerKey' >> beam.combiners.Count.PerKey()
            | 'FilterNull' >> beam.Filter(lambda x: x[0])
            | 'Values' >> beam.Values()
            | 'FindMinimum' >> beam.CombineGlobally(lambda elements: min(elements or [-1]))
        )

        res = (
            data 
            | 'GroupBylabel' >> beam.GroupByKey()
            | 'Undersample' >> beam.FlatMapTuple(sample, side=beam.pvalue.AsSingleton(val))
        )
        
        _ = (
            res
            | 'ToTFExample' >> beam.Map(lambda x: convert_to_tfexample(x, schema))
            | 'Serialize' >> beam.Map(lambda x: x.SerializeToString())
            | 'WriteToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
                os.path.join(output_dir, f'Split-{split}'),
                file_name_suffix='.gz',
                num_shards=shards,
                compression_type=beam.io.filesystem.CompressionTypes.GZIP)
        )
        