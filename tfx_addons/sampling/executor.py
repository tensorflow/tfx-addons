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
"""Executor for Sampler component."""

import os
import random
from typing import Any, Dict, List, Text

import apache_beam as beam
import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.utils import io_utils, json_utils

from tfx_addons.sampling import spec


class Executor(base_beam_executor.BaseBeamExecutor):
  """Executor for Sampler."""
  def _CreatePipeline(self, unused_transform_output: Text) -> beam.Pipeline:
    """Creates beam pipeline.
    Args:
      unused_transform_output: unused.
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
          other non-sampled spits, as specified by null_classes.
      exec_properties: A dict of execution properties, including:
        - name: Optional unique name. Necessary if multiple components are
          declared in the same pipeline.
        - label: The name of the column containing class names to sample by.
        - splits: A list containing splits to sample.
        - copy_others: Determines whether we copy over the splits that aren't
          sampled, or just exclude them from the output artifact. Defaults
          to True.
        - shards: The number of files that each sampled split should
          contain. Default 0 is Beam's tfrecordio function's default.
        - null_classes: A list determining which classes that we should
          not sample. Defaults to None.
        - sampling_strategy: An enum of type SamplingStrategy, determining if
          the executor should over or undersample.
    Returns:
      None
    """

    self._log_startup(input_dict, output_dict, exec_properties)

    label = exec_properties[spec.SAMPLER_LABEL_KEY]
    sampling_strategy = exec_properties[spec.SAMPLER_SAMPLE_KEY]
    splits = json_utils.loads(exec_properties[spec.SAMPLER_SPLIT_KEY])
    copy_others = exec_properties[spec.SAMPLER_COPY_KEY]
    shards = exec_properties[spec.SAMPLER_SHARDS_KEY]
    null_classes = json_utils.loads(exec_properties[spec.SAMPLER_CLASSES_KEY])

    input_artifact = artifact_utils.get_single_instance(
        input_dict[spec.SAMPLER_INPUT_KEY])
    output_artifact = artifact_utils.get_single_instance(
        output_dict[spec.SAMPLER_OUTPUT_KEY])

    if sampling_strategy not in (spec.SamplingStrategy.UNDERSAMPLE,
                                 spec.SamplingStrategy.OVERSAMPLE):
      raise ValueError("Invalid sampling strategy!")

    if not splits:
      raise ValueError("No splits are marked for sampling!")

    for split in splits:
      if split not in input_artifact.split_names:
        raise ValueError(
            f"Invalid split name {split} is not in input artifact!")

    if shards < 0:
      raise ValueError("Shards value must be non-negative!")

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
        split_dir = os.path.join(output_dir, f"Split-{split}")
        with self._CreatePipeline(split_dir) as p:
          data = read_tfexamples(p, uri, label)
          merged = sample_examples(data, null_classes, sampling_strategy)
          write_tfexamples(merged, shards, split_dir)
      elif copy_others:  # Copy the other split if copy_others is True
        input_dir = uri
        output_dir = artifact_utils.get_split_uri([output_artifact], split)
        for filename in fileio.listdir(input_dir):
          input_uri = os.path.join(input_dir, filename)
          output_uri = os.path.join(output_dir, filename)
          io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)


def _generate_elements(example, label):
  """Function that fetches the class label from a tf.Example and returns one
  item in a K-V PCollection with the key as the label and the value as the
  string-parsed tf.Example.

  Args:
    example: a tf.Example in serialized format, taken directly from a
      TFRecordDataset.
    label: string containing the name of the categorical variable that we are
      extracting from the example.
  Returns:
    Tuple with two items. First item is a class label; second item is the input
      tf.Example, deserialized and parsed from string format.
  """

  class_label = None
  parsed = tf.train.Example.FromString(example.numpy())
  if parsed.features.feature[label].int64_list.value:
    val = parsed.features.feature[label].int64_list.value
    if len(val) > 0:
      class_label = val[0]
  else:
    val = parsed.features.feature[label].bytes_list.value
    if len(val) > 0:
      class_label = val[0].decode()
  return (class_label, parsed)


def sample_data(_,
                val,
                sampling_strategy=spec.SamplingStrategy.UNDERSAMPLE,
                side=0):
  """Function called in a Beam pipeline that performs sampling using Python's
  random module on an input key:value pair, where the key is the class label
  and the values are the data points to sample. Note that the key is discarded."""

  if sampling_strategy == spec.SamplingStrategy.UNDERSAMPLE:
    random_sample_data = random.sample(val, side)
  elif sampling_strategy == spec.SamplingStrategy.OVERSAMPLE:
    random_sample_data = random.choices(val, k=side)
  else:
    raise ValueError("Invalid value for sampling_strategy variable!")

  for item in random_sample_data:
    yield item


def filter_null(item, keep_null=False, null_vals=None):
  """Function that returns or doesn't return the inputted item if its first
  value is either a False value or is in the inputted null_vals list.

  This function first determines if the item's first value is equivalent to
  False using bool(), with one exception; 0 is considered as "True". If the
  first value is in null_vals, the first value is automatically considered a
  "null value" and is therefore considered to be False. If the value is False,
  then None is returned; if the value is True, then the original item is
  returned. The keep_null value reverses this, so True values return None,
  and False values return the item.

  Args:
    item: Tuple whose first value determines whether it is returned or not.
      Should always be a two-value tuple, with the first value being the class
      value and the second value being all examples that belong to that class.
    keep_null: Determines whether we keep False/"null" values or True/not
      "null" values.
    null_vals: List containing values that should be considered as False/"null".
  Returns:
    None or the inputted item, depending on if the item is False/in null_vals,
      and then depending on the value of keep_null.
  """

  if item[0] == 0:
    keep = True
  else:
    keep = bool(item[0])

  if null_vals and str(item[0]) in null_vals and keep:
    keep = False
  keep ^= keep_null

  return item if keep else None


def read_tfexamples(p, uri, label):
  """Function that reads tf.Examples from tfRecord files and converts them
  to a K-V PCollection usable by Beam."""

  dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(f'{uri}/*'),
                                    compression_type="GZIP")

  # Take the input TFRecordDataset and extract the class label that we want.
  # Output format is a K-V PCollection: {class_label: TFRecord in string format}
  data = (p
          | "DatasetToPCollection" >> beam.Create(dataset)
          | "MapToLabel" >> beam.Map(_generate_elements, label))
  return data


def sample_examples(data, null_classes, sampling_strategy):
  """Function that performs the sampling given a label-mapped dataset."""

  # Finds the minimum frequency of all classes in the input label.
  # Output is a singleton PCollection with the minimum # of examples.

  def find_minimum(elements):
    return min(elements or [0])

  def find_maximum(elements):
    return max(elements or [0])

  if sampling_strategy == spec.SamplingStrategy.UNDERSAMPLE:
    sample_fn = find_minimum
  elif sampling_strategy == spec.SamplingStrategy.OVERSAMPLE:
    sample_fn = find_maximum
  else:
    raise ValueError("Invalid value for sampling_strategy variable!")

  val = (data
         | "CountPerKey" >> beam.combiners.Count.PerKey()
         | "FilterNullCount" >>
         beam.Filter(lambda x: filter_null(x, null_vals=null_classes))
         | "Values" >> beam.Values()
         | "GetSample" >> beam.CombineGlobally(sample_fn))

  # Actually performs the undersampling functionality.
  # Output format is a K-V PCollection: {class_label: TFRecord in string format}
  res = (data
         | "GroupBylabel" >> beam.GroupByKey()
         | "FilterNull" >>
         beam.Filter(lambda x: filter_null(x, null_vals=null_classes))
         | "Sample" >> beam.FlatMapTuple(sample_data,
                                         sampling_strategy=sampling_strategy,
                                         side=beam.pvalue.AsSingleton(val)))

  # Take out all the null values from the beginning and put them back in the pipeline
  null = (data
          | "ExtractNull" >> beam.Filter(
              lambda x: filter_null(x, keep_null=True, null_vals=null_classes))
          | "NullValues" >> beam.Values())
  return (res, null) | "Merge PCollections" >> beam.Flatten()


def write_tfexamples(examples, shards, output_dir):
  # Write the final set of TFRecords to the output artifact's files.
  _ = (examples
       | "Serialize" >> beam.Map(lambda x: x.SerializeToString())
       | "WriteToTFRecord" >> beam.io.tfrecordio.WriteToTFRecord(
           output_dir,
           file_name_suffix=".gz",
           num_shards=shards,
           compression_type=beam.io.filesystem.CompressionTypes.GZIP,
       ))
