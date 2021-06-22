# TODO: fix these imports and simplify them (only import what we have to from everything instead of full modules), same for other files
import json
import os
import numpy as np
from typing import Any, Dict, List, Text
import tensorflow as tf

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.utils import io_utils

from tfx import v1 as tfx
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.components.util import tfxio_utils
from tfx.dsl.component.experimental.decorators import component
import apache_beam as beam
import random

class UndersamplingExecutor(base_executor.BaseExecutor):
  """Executor for UndersamplingComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
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
    splits = exec_properties["splits"]

    input_artifact = artifact_utils.get_single_instance(
        input_dict['input_data'])
    output_artifact = artifact_utils.get_single_instance(
        output_dict['output_data'])
    output_artifact.split_names = input_artifact.split_names

    split_to_instance = {}

    for split in json.loads(input_artifact.split_names):
      uri = artifact_utils.get_split_uri([input_artifact], split)
      split_to_instance[split] = uri

    # TODO: refactor the below code
    tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(examples=[input_artifact], telemetry_descriptors=[])
    split_and_tfxio = [(split, tfxio_factory(io_utils.all_files_pattern(artifact_utils.get_split_uri([input_artifact], split))))
                    for split in artifact_utils.decode_split_names(input_artifact.split_names)[:1]]
    
    # TODO: add option to undersample more splits
    self.undersample(split_and_tfxio[0], artifact_utils.get_split_uri([output_artifact], split))
    
    # just copy the rest of the data that isn't split
    for split, instance in split_to_instance.items():
      if not split in splits:
        input_dir = instance
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        os.mkdir(output_dir)
        for filename in fileio.listdir(input_dir):
          input_uri = os.path.join(input_dir, filename)
          output_uri = os.path.join(output_dir, filename)
          io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)

  def undersample(self, input_data, output_dir):
    def generate_elements(data):
        for i in range(len(data[list(data.keys())[0]])):
            yield {key: data[key][i][0] if len(data[key][i]) > 0 else "" for key in data.keys()}

    def sample(key, value, side=0):
        for item in random.sample(value, side):
            yield item
            
    def convert_to_tfexample(data):
        features = dict()
        for key, val in data.items():
            if isinstance(val, (int, np.integer)):
                features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data[key]]))
            elif isinstance(val, (float, np.inexact)):
                features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[data[key]]))
            elif isinstance(val, str):
                features[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(data[key])]))
            else:
                features[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[key]]))
        return tf.train.Example(features=tf.train.Features(feature=features))

    split, tfxio = input_data

    with beam.Pipeline() as p:
        data = (
            # TODO: convert to list and back using a schema to save key space?
            # TODO: name every single one of the pipeline segments
            # TODO: input the actual label instead of "company" placeholder below
            p 
            | 'TFXIORead[%s]' % split >> tfxio.BeamSource()
            | 'DictConversion' >> beam.Map(lambda x: x.to_pydict())
            | 'ConversionCleanup' >> beam.FlatMap(generate_elements)
            | 'MapToLabel' >> beam.Map(lambda x: (x["company"], x))
        )

        val = (
            data
            | 'CountPerKey' >> beam.combiners.Count.PerKey()
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
            | 'ToTFExample' >> beam.Map(lambda x: convert_to_tfexample(x))
            | 'Serialize' >> beam.Map(lambda x: x.SerializeToString())
            # TODO: add the options for multiple files (same as the original artifact?)
            | 'WriteToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
                output_dir,
                compression_type=beam.io.filesystem.CompressionTypes.GZIP)
        )
        