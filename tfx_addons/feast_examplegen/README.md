# FeastExampleGen

ExampleGen for Feast feature store.

This component generates a Dataset out of a Feast entity_query and either a list of features or a feature service key.

## Installation

```sh
pip install tfx-addons[feast_examplegen]
```

## Example usage

```python
example_gen = FeastExampleGen(
  repo_config=RepoConfig(register="gs://..."),
  entity_query="SELECT user, timestamp from some_user_dataset",
  features=["f1", "f2"],
)
```
Component can be configured the same way as any [QueryBasedExampleGen](https://www.tensorflow.org/tfx/guide/examplegen#query-based_examplegen_customization_experimental).

Component `outputs` contains:
   - `examples`: Channel of type `standard_artifacts.Examples` for output train
                 and eval examples.

## Extra information

- [Proposal](https://github.com/tensorflow/tfx-addons/blob/main/proposals/20210525-feast_example_gen.md)
- [Example usage](https://github.com/tensorflow/tfx-addons/tree/main/examples/fraud_feast)
