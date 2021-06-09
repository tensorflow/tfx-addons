#### SIG TFX-Addons
# Project Proposal

**Your name:** Ryan Clough

**Your email:** rclough@spotify.com

**Your company/organization:** Spotify

**Project name:** Example Filter

## Project Description
Beam based component that can filter Examples based on a user-defined predicate function.

## Project Category
Choose 1: Component

## Project Use-Case(s)
Data can be imported into TFX in a number of ways, and indeed, sometimes the dataset you wish to load is not under your direct 
control. In cases like these, it is useful to have a component that can filter your input data with simple rules. Ex: filter 
all records where `feature_a >= 1`. 

Our organization currently has a component for this purpose that is in active use. It is not as robust as it could be.

It is also worth conidering that we may wish to try and promote this functionality to be included in the TFX core base ExampleGen,
so that the filtering could be done within any ExampleGen based component.

## Project Implementation
Spotify can provide the current implementation, which is based off of an old version of Tensorflow Transform. At a high level, use
of the component looks like:

```python
def predicate_fn(example)
    # Throw out Examples that used a credit card
    if b'Credit Card' in example['payment_type']:
        return False
    return True
...

filtered_examples = ExampleFilter(
    examples=examples.output,
    schema=schema.output,
    module_file=filter_module,
)
```

The current implementation is a bit dated and not so robust. It depends on a deprecated TFT proto coder, and only works on
TF Records, as it does not make use of TFXIO. As part of bringing this to TFX-addons, I think it is worth iterating on the
current design. Some initial ideas for change might be:

* Implementing it more flexibly in TFXIO
* Determine if there's a way to implement it without requiring a schema
* Making the predicate_fn operate on true data types rather than bytes (see example above)
* Adding an input that allows the user to specify splits (currently applies to all splits)

Given that it's a Beam component, I think it will have to be a fully custom component. In terms of packaging and providing,
I'm not sure the best course of action as there is not much precedent set in TFX-addons just yet.

## Project Dependencies
Current implementation uses a [proto decoder](https://github.com/tensorflow/transform/blob/v0.24.1/tensorflow_transform/coders/example_proto_coder.py#L329-L339)
deprecated from TFX 0.25 onwards, though with help we could possibly remake the componnt with TFXIO.

## Project Team
* Ryan Clough, rclough@spotify.com, @rclough
* TBD
