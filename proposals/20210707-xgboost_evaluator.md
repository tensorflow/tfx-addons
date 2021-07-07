TFX XGBoost Evaluator - Draft Project Proposal

This is the project proposal for the tfx-addons project https://github.com/tensorflow/tfx-addons/issues/13.

# Project Proposal

# Project Description
Add support for evaluating XGBoost model in the standard component Evaluator. 
Add an example pipeline that trains, evaluates and pushes an XGBoost model to CAIP.

## Project Category

Component + Example

## Project Use-Case(s)

## Project Implementation

To make the Evaluator works with XGBoost models, we can customize the Evaluator by providing a Python module with:
* `custom_eval_shared_model()` to load model artifacts that are not standard TF models
* `custom_extractors()` to inject a custom prediction extractor. Similar to (tfma.extractors.PredictExtractor)[https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/extractors/PredictExtractor], this extractor uses Beam PTransform to load and extract predictions.

### From Trainer to Evaluator: save and load XGBoost model

Option 1: working with XGBoost library directly

The XGBoost library provides a few different ways to save a model (an xgb.Booster or xgb.sklearn.XGBModel object). Backward compatibility is guaranteed in most cases. Currently, the 2 main supported formats are:
* XGBoost internal binary format. Note that Auxiliary attributes of the Python Booster object (such as feature_names) will not be loaded when using binary format.
* JSON: newer format aiming to replace the binary format

For maximum compatibility, we want the Trainer component to output both formats at the expected output directory, and can provide a helper function, which takes in a Booster object then writes `model.bin` and `model.json` to the expected directory. 
The Evaluator uses the latest version of the xgboost library to read `model.json` - this will be implemented in UDF `custom_eval_shared_model()`. This way, we can expect the loaded model object to have most necessary information retained.

Option 2: using [sklearn Pipeline](https://scikit-learn.org/stable/modules/compose.html)

```
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline

classifier = xgb.XGBClassifier(**params)
model = SkPipeline([
    ('scaler', StandardScaler()),
    ('classifier', classifier),
])
model.fit(x_train, y_train)

# you can choose to save just the XGBClassifier:
model.steps[1][1].save_model(...)
```
However, it’s more likely that you’ll need the whole sklearn Pipeline in downstream evaluation. There are 2 methods to save and load sklearn Pipeline object:

Using joblib:
```
import joblib
joblib.dump(pipeline, 'model.joblib')
```
Note that CAIP asks users to use `sklearn.externals.joblib` rather than the bare `joblib`, but newer versions of sklearn have deprecated `skearn.externals`.

Using pickle:
```
import pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
```
The main downside of working with sklearn Pipeline is potentially losing portability, this will be discussed further in the summary section.


### Custom Extractor for the Evaluator

The Evaluator component will also utilize a custom prediction extractor, which would load and run our EvalSharedModel(s) on given examples. Xgboost models cannot accept `tf.Example`s as input, so they will have to be converted within the function.

Our custom prediction extractor essentially governs conversion of data to formats that xgboost can accept, extraction of the necessary features of the data, the actual prediction, and framework code supporting all of these operations. It will be passed into (this)[https://tensorflow.google.cn/tfx/model_analysis/api_docs/python/tfma/default_extractors] `tfma.default_extractors` function for use in the Evaluator.

Currently, we plan to support running the Evaluator with Apache Beam through the use of a customized prediction DoFn to load, process, and run predictions on models, and a simple pipeline wrapper that calls this function on extracts.

The actual implementation of the custom prediction extractor depends on whether it should work with a native XGBoost serialized model (option 1 from above).
Pros:
Universal among the various XGBoost interfaces (Python, JVM, C++, etc.)
Some level of backward compatibility is guaranteed
Still retain attributes such as feature_names, feature_types, etc. (in newer xgboost versions)

Another option is a pickled sklearn Pipeline (option 2 from above). 
Pros:
Another wrapping layer means more flexibility, you can add some pre-processing and post-processing to the sklearn Pipeline, try out other types of models, etc.
Most of the code needed for sklearn-compatible Trainer and Evaluator in the penguin sklearn pipeline can be reused
Cons:
Extra dependency on sklearn
Using Python pickle standard library or joblib, which is specific to Python
Lack of guarantee for backward compatibility

By default, we plan to go with option 1 for simplicity and consistency.

Open questions:
From training performance view point, is there a difference between using native xgboost vs using sklearn Pipeline?

### Testing and Example Pipeline

In the same spirit as https://github.com/tensorflow/tfx-addons/blob/main/proposals/20210404-sklearn_example.md, we will add an example pipeline that runs locally or on GCP using Dataflow and Kubeflow Pipelines. This pipeline will have its end to end local unit test.

The model can be pushed to CAIP. CAIP runtime version 2.5 runs XGBoost 1.4.0.

This example pipeline will not be packaged, instead, users just need to clone the source code to run the example.

## Project Dependencies
* `xgboost>=1.4.0`
* `sklearn>=?`

## References

* [TFX Penguin sklearn example pipeline](https://github.com/tensorflow/tfx-addons/tree/main/projects/examples/sklearn_penguins)
* [CAIP Exporting models for prediction](https://cloud.google.com/ai-platform/prediction/docs/exporting-for-prediction)

## Project Team

Daniel Kim, kindalime, danielk@twitter.com
Vincent Nguyen, cent5, vincentn@twitter.com
