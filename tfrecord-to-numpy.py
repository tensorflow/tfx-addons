# Reading a TFRecord file
# Serialized tensors can be easily parsed using `tf.train.Example.ParseFromString`

import tensorflow as tf
import numpy as np

filename = "something"
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)

for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)

# That returns a `tf.train.Example` proto which is dificult to use as is, but it's fundamentally a representation of a:
#
# Dict[str,
#   Union[List[float],
#        List[int],
#        List[str]]]
#
# The following code manually converts the `Example` to a dictionary of NumPy arrays, without using tensorflow Ops.
# Refer to [the PROTO file](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto) for details.

result = {}
# example.features.feature is the dictionary
for key, feature in example.features.feature.items():
  # The values are the Feature objects which contain a `kind` which contains:
  # one of three fields: bytes_list, float_list, int64_list

  kind = feature.WhichOneof('kind')
  result[key] = np.array(getattr(feature, kind).value)

print(result)