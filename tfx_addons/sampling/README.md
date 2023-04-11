# Sampler component

A TFX component to sample examples.

The sampling component wraps an Apache Beam pipeline to process
data in an TFX pipeline. This component loads in tf.Record files from
an earlier example artifact, processes the 'train' split by default,
samples the split by a given label's classes, and stores the new
set of sampled examples into its own example artifact in
tf.Record format.

The sampling is probabilistic estimation. Note that in small datasets
this may result in worse datasets or such. This module is meant to
approximate sampling using probability.

By default, the component will ignore all examples with a null value
(more precisely, a value that evaluates to False) for the given label,
although more values can be added in as necessary. Additionally, it will
copy all non-'train' splits, through this behavior can be changed as well.
The component will save the examples in a user-specified number of files,
and it can be given a name as well.

## Example usage
```
import tfx_addons as tfxa

under = tfxa.sampling.Sampler(
  examples=example_gen.outputs['examples'],
  sampling_strategy=tfxa.sampling.SamplingStrategy.UNDERSAMPLE
)
```

Component `outputs` contains:
  - `sampled_examples`: Channel of type `standard_artifacts.Examples` for
    materialized sampled examples, based on the input splits, which includes
    copied splits unless otherwise specified by copy_others.

[Initial Proposal](https://github.com/tensorflow/tfx-addons/blob/main/proposals/20210721-sampling_component.md)
