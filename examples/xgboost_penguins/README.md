# Penguin Classification XGBoost Example

Expanded the [TFX penguin example
pipeline](https://github.com/tensorflow/tfx/tree/master/tfx/examples/penguin)
and use [xgboost](https://xgboost.readthedocs.io/en/latest/)
to build and train the model.

Also see [XGBoost Evaluator](/tfx_addons/xgboost_evaluator/README.md) for more
context on how the trained model can evaluated.

## Local Example
Execute the pipeline python file. Output can be found at `~/tfx`:

```
python examples/xgboost_penguins/penguin_pipeline_local.py
```

## Run e2e test

```
pip install -e ".[all,test]"`
pytest examples/xgboost_penguins
```