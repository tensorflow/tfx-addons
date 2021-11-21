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
"""Executor test for the sampling component's executor."""

import filecmp
import os
import random
import tempfile

import apache_beam as beam
import tensorflow as tf
from absl.testing import absltest
from apache_beam.testing.util import assert_that, equal_to
from tfx.components.util import tfxio_utils
from tfx.dsl.io import fileio
from tfx.types import artifact_utils, standard_artifacts
from tfx.utils import io_utils, json_utils

from tfx_addons.sampling import executor, spec


class ExecutorTest(absltest.TestCase):
  def _validate_output(self, output, splits, num=1):
    def generate_elements(data):
      for i in range(len(data[list(data.keys())[0]])):
        gen = dict()
        for key in data.keys():
          if data[key][i] and len(data[key][i]) > 0:
            gen[key] = data[key][i][0]
          else:
            gen[key] = ""
        yield gen

    tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(
        examples=[output], telemetry_descriptors=[])
    for split in splits:
      tfxio = tfxio_factory(
          io_utils.all_files_pattern(
              artifact_utils.get_split_uri([output], split)))

      with beam.Pipeline() as p:
        data = (p
                | 'TFXIORead[%s]' % split >> tfxio.BeamSource()
                | 'DictConversion' >> beam.Map(lambda x: x.to_pydict())
                | 'ConversionCleanup' >> beam.FlatMap(generate_elements)
                | 'MapToLabel' >> beam.Map(lambda x: (x['label'], x))
                | 'CountPerKey' >> beam.combiners.Count.PerKey()
                | 'FilterNull' >> beam.Filter(lambda x: x[0])
                | 'Values' >> beam.Values()
                | 'Distinct' >> beam.Distinct()
                | 'Count' >> beam.combiners.Count.Globally())

    assert_that(data, equal_to([num]))

  def _validate_same(self, dir0, dir1):
    comp = filecmp.dircmp(dir0, dir1)
    self.assertTrue(comp.left_only == [])
    self.assertTrue(comp.right_only == [])
    self.assertTrue(comp.diff_files == [])

  def _run_exec(self, exec_properties):
    source_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', tempfile.mkdtemp()),
        self._testMethodName)
    fileio.makedirs(output_data_dir)

    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, "example_gen")
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    input_dict = {
        spec.SAMPLER_INPUT_KEY: [examples],
    }

    # Create output dict.
    output = standard_artifacts.Examples()
    output.uri = output_data_dir
    output_dict = {
        spec.SAMPLER_OUTPUT_KEY: [output],
    }

    # Run executor.
    under = executor.Executor()
    under.Do(input_dict, output_dict, exec_properties)

    return output

  def testDo(self):
    exec_properties = {
        spec.SAMPLER_LABEL_KEY:
        'label',
        spec.SAMPLER_NAME_KEY:
        'undersampling',
        spec.SAMPLER_SPLIT_KEY:
        json_utils.dumps(['train']),
        # List needs to be serialized before being passed into Do function.
        spec.SAMPLER_COPY_KEY:
        True,
        spec.SAMPLER_SHARDS_KEY:
        1,
        spec.SAMPLER_CLASSES_KEY:
        json_utils.dumps([]),
        spec.SAMPLER_SAMPLE_KEY:
        True,
    }

    output = self._run_exec(exec_properties)

    # Check outputs.
    self._validate_output(output, ['train'])
    self.assertTrue(fileio.exists(os.path.join(output.uri, 'Split-train')))
    self.assertTrue(fileio.exists(os.path.join(output.uri, 'Split-eval')))

  def testKeepClasses(self):
    exec_properties = {
        spec.SAMPLER_LABEL_KEY:
        'label',
        spec.SAMPLER_NAME_KEY:
        'undersampling',
        spec.SAMPLER_SPLIT_KEY:
        json_utils.dumps(['train']),
        # List needs to be serialized before being passed into Do function.
        spec.SAMPLER_COPY_KEY:
        True,
        spec.SAMPLER_SHARDS_KEY:
        1,
        spec.SAMPLER_CLASSES_KEY:
        json_utils.dumps(['None']),
        spec.SAMPLER_SAMPLE_KEY:
        spec.SamplingStrategy.UNDERSAMPLE,
    }

    output = self._run_exec(exec_properties)

    # Check outputs.
    self._validate_output(output, ['train'], 2)

  def testShards(self):
    exec_properties = {
        spec.SAMPLER_LABEL_KEY: 'label',
        spec.SAMPLER_NAME_KEY: 'undersampling',
        spec.SAMPLER_SPLIT_KEY: json_utils.dumps(
            ['train']
        ),  # List needs to be serialized before being passed into Do function.
        spec.SAMPLER_COPY_KEY: True,
        spec.SAMPLER_SHARDS_KEY: 20,
        spec.SAMPLER_CLASSES_KEY: json_utils.dumps([]),
        spec.SAMPLER_SAMPLE_KEY: spec.SamplingStrategy.UNDERSAMPLE,
    }

    output = self._run_exec(exec_properties)

    # Check outputs.
    out = os.path.join(output.uri, 'Split-train')
    self.assertTrue(
        len([
            name for name in os.listdir(out)
            if os.path.isfile(os.path.join(out, name))
        ]) == 20)

  def testSplits(self):
    exec_properties = {
        spec.SAMPLER_LABEL_KEY: 'label',
        spec.SAMPLER_NAME_KEY: 'undersampling',
        spec.SAMPLER_SPLIT_KEY: json_utils.dumps(
            ['train', 'eval']
        ),  # List needs to be serialized before being passed into Do function.
        spec.SAMPLER_COPY_KEY: True,
        spec.SAMPLER_SHARDS_KEY: 1,
        spec.SAMPLER_CLASSES_KEY: json_utils.dumps([]),
        spec.SAMPLER_SAMPLE_KEY: spec.SamplingStrategy.UNDERSAMPLE,
    }

    output = self._run_exec(exec_properties)

    # Check outputs.
    self._validate_output(output, ['train'])
    self._validate_output(output, ['eval'])

  def testCopy(self):
    exec_properties = {
        spec.SAMPLER_LABEL_KEY: 'label',
        spec.SAMPLER_NAME_KEY: 'undersampling',
        spec.SAMPLER_SPLIT_KEY: json_utils.dumps(
            ['train']
        ),  # List needs to be serialized before being passed into Do function.
        spec.SAMPLER_COPY_KEY: False,
        spec.SAMPLER_SHARDS_KEY: 1,
        spec.SAMPLER_CLASSES_KEY: json_utils.dumps([]),
        spec.SAMPLER_SAMPLE_KEY: spec.SamplingStrategy.UNDERSAMPLE,
    }

    output = self._run_exec(exec_properties)

    self.assertFalse(fileio.exists(os.path.join(output.uri, 'Split-eval')))

  # Pipeline tests below!

  def testFilter(self):
    assert executor.filter_null([5, 5])  # return
    assert executor.filter_null([0, 0])  # return
    assert not executor.filter_null(["", ""])  # no return
    assert not executor.filter_null(["", 5])  # no return
    assert not executor.filter_null([5, 5], keep_null=True)  # no return
    assert not executor.filter_null([0, 0], keep_null=True)  # no return
    assert executor.filter_null(["", ""], keep_null=True)  # return
    assert not executor.filter_null([5, 5], null_vals=["5"])  # no return
    assert executor.filter_null([5, 5], keep_null=True,
                                null_vals=["5"])  # return
    assert executor.filter_null(["", ""], keep_null=True,
                                null_vals=["5"])  # return

  def testPipelineMin(self):
    random.seed(0)
    dataset = [("1", 1), ("1", 1), ("1", 1), ("2", 2), ("2", 2), ("2", 2),
               ("2", 2), ("3", 3), ("3", 3), ("", 0)]
    expected = [1, 1, 2, 2, 3, 3, 0]

    with beam.Pipeline() as p:
      data = p | beam.Create(dataset)
      merged = executor.sample_examples(data, None,
                                        spec.SamplingStrategy.UNDERSAMPLE)
      assert_that(merged, equal_to(expected))

  def testPipelineMax(self):
    random.seed(0)
    dataset = [("1", 1), ("1", 1), ("1", 1), ("2", 2), ("2", 2), ("2", 2),
               ("2", 2), ("3", 3), ("3", 3), ("", 0)]
    expected = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0]

    with beam.Pipeline() as p:
      data = p | beam.Create(dataset)
      merged = executor.sample_examples(data, None,
                                        spec.SamplingStrategy.OVERSAMPLE)
      assert_that(merged, equal_to(expected))

  def testMinimum(self):
    dataset = [("1", 1), ("1", 1), ("1", 1), ("2", 2), ("2", 2), ("2", 2),
               ("2", 2), ("3", 3), ("3", 3), ("", 0)]

    def find_minimum(elements):
      return min(elements or [0])

    with beam.Pipeline() as p:
      val = (p
             | beam.Create(dataset)
             | "CountPerKey" >> beam.combiners.Count.PerKey()
             | "FilterNullCount" >>
             beam.Filter(lambda x: executor.filter_null(x, null_vals=None))
             | "Values" >> beam.Values()
             | "GetSample" >> beam.CombineGlobally(find_minimum))
      assert_that(val, equal_to([2]))

  def testMaximum(self):
    dataset = [("1", 1), ("1", 1), ("1", 1), ("2", 2), ("2", 2), ("2", 2),
               ("2", 2), ("3", 3), ("3", 3), ("", 0)]

    def find_maximum(elements):
      return max(elements or [0])

    with beam.Pipeline() as p:
      val = (p
             | beam.Create(dataset)
             | "CountPerKey" >> beam.combiners.Count.PerKey()
             | "FilterNullCount" >>
             beam.Filter(lambda x: executor.filter_null(x, null_vals=None))
             | "Values" >> beam.Values()
             | "GetSample" >> beam.CombineGlobally(find_maximum))
      assert_that(val, equal_to([4]))

  def testException(self):
    exec_properties = {
        spec.SAMPLER_LABEL_KEY: 'label',
        spec.SAMPLER_NAME_KEY: 'undersampling',
        spec.SAMPLER_SPLIT_KEY: json_utils.dumps(
            ['bad']
        ),  # List needs to be serialized before being passed into Do function.
        spec.SAMPLER_COPY_KEY: False,
        spec.SAMPLER_SHARDS_KEY: 1,
        spec.SAMPLER_CLASSES_KEY: json_utils.dumps([]),
        spec.SAMPLER_SAMPLE_KEY: spec.SamplingStrategy.UNDERSAMPLE,
    }

    with self.assertRaises(ValueError) as context:
      self._run_exec(exec_properties)
    self.assertTrue('Invalid split name bad is not in input artifact!' in str(
        context.exception))


if __name__ == '__main__':
  tf.test.main()
