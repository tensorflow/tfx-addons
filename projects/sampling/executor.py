"""Executor for TensorFlow Transform."""

import os
import tensorflow as tf
import random
import apache_beam as beam
from typing import Any, Dict, List, Text

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.components.util import tfxio_utils
from tfx.utils import json_utils


class Executor(base_executor.BaseExecutor):
  """Executor for Undersample."""

  def Do(
    self,
    input_dict: Dict[Text, List[types.Artifact]],
    output_dict: Dict[Text, List[types.Artifact]],
    exec_properties: Dict[Text, Any],
  ) -> None:

    """Undersample executor entrypoint.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
      - examples: A list of type `standard_artifacts.Examples` which should
      contain custom splits specified in splits_config. If custom split is
      not provided, this should contain two splits 'train' and 'eval'.
      output_dict: Output dict from key to a list of artifacts, including:
      - undersampled_examples: Undersampled examples, only for the given
      splits as specified in splits. May also include copies of the
      other non-undersampled spits, as specified by keep_classes.
      exec_properties: A dict of execution properties, including:
      - name: Optional unique name. Necessary if multiple components are
      declared in the same pipeline.
      - label: The name of the column containing class names to undersample by.
      - splits: A list containing splits to undersample. Defaults to ['train'].
      - copy_others: Determines whether we copy over the splits that aren't
      undersampled, or just exclude them from the output artifact. Defualts
      to True.
      - shards: The number of files that each undersampled split should
      contain. Default 0 is Beam's tfrecordio function's default.
      - keep_classes: A list determining which classes that we should
      not undersample. Defaults to None.

    Returns:
      None
    """

    self._log_startup(input_dict, output_dict, exec_properties)
    label = exec_properties["label"]
    splits = json_utils.loads(exec_properties["splits"])
    copy_others = exec_properties["copy_others"]
    shards = exec_properties["shards"]
    keep_classes = json_utils.loads(exec_properties["keep_classes"])

    input_artifact = artifact_utils.get_single_instance(input_dict["input_data"])
    output_artifact = artifact_utils.get_single_instance(output_dict["output_data"])

    if copy_others:
      output_artifact.split_names = input_artifact.split_names
    else:
      output_artifact.split_names = splits

    # Fetch the input uri for each split
    split_data = {}
    for split in artifact_utils.decode_split_names(input_artifact.split_names):
      uri = artifact_utils.get_split_uri([input_artifact], split)
      split_data[split] = uri

    for split, uri in split_data.items():
      if split in splits:  # Undersampling split
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        os.mkdir(output_dir)
        self.undersample(uri, label, shards, keep_classes, os.path.join(output_dir, f"Split-{split}"),)
      elif copy_others:  # Copy the other split if copy_others is True
        input_dir = uri
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        os.mkdir(output_dir)
        for filename in fileio.listdir(input_dir):
          input_uri = os.path.join(input_dir, filename)
          output_uri = os.path.join(output_dir, filename)
          io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)

  def undersample(self, uri, label, shards, keep_classes, output_dir):
    """Function that actually undersamples the given split.

    Args:
      uri: The input uri for the specific split of the input example artifact.
      label: The name of the column containing class names to undersample by.
      shards: The number of files that each undersampled split should
      contain. Default 0 is Beam's tfrecordio function's default.
      keep_classes: A list determining which classes that we should
      not undersample. Defaults to None.
      output_dir: The output directory for the split of the output artifact.

    Returns:
      None
    """

    def generate_elements(x):
      parsed = tf.train.Example.FromString(x.numpy())
      val = parsed.features.feature[label].bytes_list.value
      if len(val) > 0:
        class_label = val[0].decode()
      else:
        class_label = None
      return (class_label, parsed)

    def sample(key, value, side=0):
      for item in random.sample(value, side):
        yield item

    def filter_null(item, keep_null=False, null_vals=None):
      keep = not item[0]  # Note that 0 is included in this filter!
      if null_vals:
        keep = keep or str(item[0]) in null_vals
      keep ^= not keep_null  # null is True if we want to keep nulls
      if keep:
        return item[1]

    files = [os.path.join(uri, name) for name in os.listdir(uri)]
    dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")

    with beam.Pipeline() as p:
      # Take the input TFRecordDataset and extract the class label that we want.
      # Output format is a K-V PCollection: {class_label: TFRecord in string format}
      data = (
        p
        | "DatasetToPCollection" >> beam.Create(dataset)
        | "MapToLabel" >> beam.Map(generate_elements)
      )

      # Finds the minimum frequency of all classes in the input label.
      # Output is a singleton PCollection with the minimum # of examples.
      val = beam.pvalue.AsSingleton(
        (
          data
          | "CountPerKey" >> beam.combiners.Count.PerKey()
          | "FilterNull" >> beam.Filter(lambda x: filter_null(x, null_vals=keep_classes))
          | "Values" >> beam.Values()
          | "FindMinimum" >> beam.CombineGlobally(lambda elements: min(elements or [-1]))
        )
      )

      # Actually performs the undersampling functionality.
      # Output format is a K-V PCollection: {class_label: TFRecord in string format}
      res = (
        data
        | "GroupBylabel" >> beam.GroupByKey()
        | "Undersample" >> beam.FlatMapTuple(sample, side=val)
      )

      # Take out all the null values from the beginning and put them back in the pipeline
      null = (
        data
        | "ExtractNull">> beam.Filter(lambda x: filter_null(x, keep_null=True, null_vals=keep_classes))
        | "NullValues" >> beam.Values()
      )
      merged = (res, null) | "Merge PCollections" >> beam.Flatten()

      # Write the final set of TFRecords to the output artifact's files.
      _ = (
        merged
        | "Serialize" >> beam.Map(lambda x: x.SerializeToString())
        | "WriteToTFRecord" >> beam.io.tfrecordio.WriteToTFRecord(
          output_dir,
          file_name_suffix=".gz",
          num_shards=shards,
          compression_type=beam.io.filesystem.CompressionTypes.GZIP,
        )
      )
