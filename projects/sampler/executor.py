"""Executor for TensorFlow Transform."""

import os
import tensorflow as tf
import random
import apache_beam as beam
from typing import Any, Dict, List, Text

from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.utils import json_utils
from tfx.utils import io_utils

class Executor(base_beam_executor.BaseBeamExecutor):
  """Executor for Sampler."""

  def _CreatePipeline(
      self, unused_transform_output_path: Text) -> beam.Pipeline:
    """Creates beam pipeline.
    Args:
      unused_transform_output_path: unused.
    Returns:
      Beam pipeline.
    """
    return self._make_beam_pipeline()

  def Do(
    self,
    input_dict: Dict[Text, List[types.Artifact]],
    output_dict: Dict[Text, List[types.Artifact]],
    exec_properties: Dict[Text, Any],
  ) -> None:

    """Sampler executor entrypoint.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
      - examples: A list of type `standard_artifacts.Examples` which should
      contain custom splits specified in splits_config. If custom split is
      not provided, this should contain two splits 'train' and 'eval'.
      output_dict: Output dict from key to a list of artifacts, including:
      - sampled_examples: sampled examples, only for the given
      splits as specified in splits. May also include copies of the
      other non-sampled spits, as specified by keep_classes.
      exec_properties: A dict of execution properties, including:
      - name: Optional unique name. Necessary if multiple components are
      declared in the same pipeline.
      - label: The name of the column containing class names to sample by.
      - splits: A list containing splits to sample. Defaults to ['train'].
      - copy_others: Determines whether we copy over the splits that aren't
      sampled, or just exclude them from the output artifact. Defualts
      to True.
      - shards: The number of files that each sampled split should
      contain. Default 0 is Beam's tfrecordio function's default.
      - keep_classes: A list determining which classes that we should
      not sample. Defaults to None.

    Returns:
      None
    """

    self._log_startup(input_dict, output_dict, exec_properties)
    label = exec_properties['label']
    undersample = exec_properties['undersample']
    splits = json_utils.loads(exec_properties['splits'])
    copy_others = exec_properties['copy_others']
    shards = exec_properties['shards']
    keep_classes = json_utils.loads(exec_properties['keep_classes'])

    input_artifact = artifact_utils.get_single_instance(input_dict['input_data'])
    output_artifact = artifact_utils.get_single_instance(output_dict['output_data'])

    if copy_others:
      output_artifact.split_names = input_artifact.split_names
    else:
      output_artifact.split_names = artifact_utils.encode_split_names(splits)

    # Fetch the input uri for each split
    split_data = {}
    for split in artifact_utils.decode_split_names(input_artifact.split_names):
      uri = artifact_utils.get_split_uri([input_artifact], split)
      split_data[split] = uri

    for split, uri in split_data.items():
      if split in splits:  # Undersampling split
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        self.sample(uri, label, shards, keep_classes, os.path.join(output_dir, f"Split-{split}"), undersample)
      elif copy_others:  # Copy the other split if copy_others is True
        input_dir = uri
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        for filename in fileio.listdir(input_dir):
          input_uri = os.path.join(input_dir, filename)
          output_uri = os.path.join(output_dir, filename)
          io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)

  def sample(self, uri, label, shards, keep_classes, output_dir, undersample):
    """Function that actually samples the given split.

    Args:
      uri: The input uri for the specific split of the input example artifact.
      label: The name of the column containing class names to sample by.
      shards: The number of files that each sampled split should
      contain. Default 0 is Beam's tfrecordio function's default.
      keep_classes: A list determining which classes that we should
      not sample. Defaults to None.
      output_dir: The output directory for the split of the output artifact.

    Returns:
      None
    """

    def generate_elements(x):
      class_label = None
      parsed = tf.train.Example.FromString(x.numpy())
      if parsed.features.feature[label].int64_list.value:
        val = parsed.features.feature[label].int64_list.value
        if len(val) > 0:
          class_label = val[0]
      else:
        val = parsed.features.feature[label].bytes_list.value
        if len(val) > 0:
          class_label = val[0].decode()
      return (class_label, parsed)

    def random_sample(key, value, side=0):
      random_sample_data = random.sample(value, side) if undersample else random.choices(value, k=side)
      for item in random_sample_data:
        yield item

    def filter_null(item, keep_null=False, null_vals=None, pr=False):
      if pr:
        print(item)
      keep = (not item[0]) or item[0] == 0  # Note that 0 is included in this filter!
      if null_vals:
        keep = keep or str(item[0]) in null_vals
      keep ^= not keep_null  # null is True if we want to keep nulls
      if keep:
        return item[1]

    def find_minimum(elements):
      return min(elements or [0])
    def find_maximum(elements):
      return max(elements or [0])

    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(f'{uri}/*'), compression_type="GZIP")

    with self._CreatePipeline(output_dir) as p:
      # Take the input TFRecordDataset and extract the class label that we want.
      # Output format is a K-V PCollection: {class_label: TFRecord in string format}
      data = (
        p
        | "DatasetToPCollection" >> beam.Create(dataset)
        | "MapToLabel" >> beam.Map(generate_elements)
      )

      # Finds the minimum frequency of all classes in the input label.
      # Output is a singleton PCollection with the minimum # of examples.

      sample_fn = find_minimum if undersample else find_maximum

      val = beam.pvalue.AsSingleton(
        (
          data
          | "CountPerKey" >> beam.combiners.Count.PerKey()
          | "FilterNull" >> beam.Filter(lambda x: filter_null(x, null_vals=keep_classes))
          | "Values" >> beam.Values()
          | "GetSample" >> beam.CombineGlobally(sample_fn)
        )
      )

      # Actually performs the undersampling functionality.
      # Output format is a K-V PCollection: {class_label: TFRecord in string format}
      res = (
        data
        | "GroupBylabel" >> beam.GroupByKey()
        | "Sample" >> beam.FlatMapTuple(random_sample, side=val)
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
