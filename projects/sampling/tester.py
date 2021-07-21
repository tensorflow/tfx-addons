import os
import tempfile
import tensorflow as tf
import filecmp
import random

import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from absl.testing import absltest

from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils
from tfx.utils import io_utils
from tfx.components.util import tfxio_utils

def sample(key, value, side=0):
  for item in random.sample(value, side):
    yield item

def filter_null(item, keep_null=False, null_vals=None, pr=False):
  if item[0] == 0:
    keep = True
  else:
    keep = not (not item[0])

  if null_vals and str(item[0]) in null_vals and keep:
    keep = False
  keep ^= keep_null
  if keep:
    return item

def testPipeline():
  random.seed(0)
  dataset = [("1", 1), ("1", 1), ("1", 1), ("2", 2), ("2", 2), ("2", 2), ("2", 2), ("3", 3), ("3", 3), ("", 0)]

  with beam.Pipeline() as p:
    data = (
      p
      | "DatasetToPCollection" >> beam.Create(dataset)
    )

    val = beam.pvalue.AsSingleton(
      (
        data
        | "CountPerKey" >> beam.combiners.Count.PerKey()
        | "FilterNullCount" >> beam.Filter(lambda x: filter_null(x))
        | "Values" >> beam.Values()
        | "FindMinimum" >> beam.CombineGlobally(lambda elements: min(elements or [-1]))
      )
    )

    res = (
      data
      | "GroupBylabel" >> beam.GroupByKey()
      | "FilterNull" >> beam.Filter(lambda x: filter_null(x))
      | "Undersample" >> beam.FlatMapTuple(sample, side=val)
    )
  
    # Take out all the null values from the beginning and put them back in the pipeline
    null = (
      data
      | "ExtractNull" >> beam.Filter(lambda x: filter_null(x, keep_null=True))
      | "NullValues" >> beam.Values()
    )

    merged = (
      (res, null) 
      | "Merge PCollections" >> beam.Flatten()
      | beam.Map(print)
    )


testPipeline()
