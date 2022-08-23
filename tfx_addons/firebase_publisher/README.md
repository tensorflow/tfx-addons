#### SIG TFX-Addons
# Project Proposal

**Your name:** Chansung Park

**Your email:** deep.diver.csp@gmail.com

**Your company/organization:** Individual(ML GDE)

**Project name:** [FirebasePublisher](https://github.com/tensorflow/tfx-addons/issues/59)

## Project Description
This project defines a custom TFX component to publish/update ML models to [Firebase ML](https://firebase.google.com/products/ml). 

## Project Category
Component

## Project Use-Case(s)
This project helps users to publish trained models directly from TFX Pusher component to Firebase ML. 

With Firebase ML, we can guarantee that mobile devices can be equipped with the latest ML model without explicitly embedding binary in the project compiling stage. We can even A/B test different versions of a model with Google Analytics when the model is published on Firebase ML.

## Project Implementation
Firebase ML Publisher component will be implemented as a Python class-based component. You can find the [actual source code](https://github.com/deep-diver/complete-mlops-system-workflow/tree/feat/firebase-publisher/training_pipeline/pipeline/components/pusher/FirebasePublisher) in my personal project. 

The implementation details
- This component behaves similar to [Pusher](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Pusher) component, but it pushes/hosts model to Firebase ML instead. To this end, FirebasePublisher interits Pusher, and it gets the following inputs

```python
FirebasePublisher(
  display_name: str,
  tags: List[str],
  storage_bucket: str,
  model: types.BaseChannel = None,
  options: Optional[Dict] = None,
  credential_path: Optional[str] = None,
  model_blessing: Optional[types.BaseChannel] = None,
)
```

- Each inputs:
  - `model` : the model from the upstream TFX component such as [Trainer](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Trainer)
  - `model_blessing` : the output of `blessing` from the Evaluator component to indicate if the given `model` is good enough to be pushed
  - `display_name` : display name to appear in Firebase ML. This should be a unique value since it will be used to search a existing model to update
  - `tags` : tags to appear in Firebase ML
  - `storage_bucket` : GCS bucket where the hosted model is stored. `gs://` should not be included
  - `credential_path` : an optional parameter, and it indicates GCS or local location where a Service Account Key (JSON) file is stored. If this parameter is not given, [Application Default Credentials](https://cloud.google.com/docs/authentication/production) will be used in GCP environment
  - `options` : additional configurations to be passed to initialize Firebase app

- It outputs the following information by the method [`_MarkPushed`](https://github.com/tensorflow/tfx/blob/3b5290aa77c2df52a4791715cfd761be7696fe81/tfx/components/pusher/executor.py#L222) from Pusher component
  - `pushed` : indicator if the model is pushed without any issue or if the model is blessed.
  - `pushed_destination` : URL string for easy access to the model from Firebase Console such as `f"https://console.firebase.google.com/u/1/project/{PROJECT_ID}/ml/custom"`
  - `pushed_version` : version string of the pushed model. This is determined in the same manner as Pusher by `str(int(time.time()))`

- The detailed behaviour of this component
  - Initialize Firebase App with `firebase_admin.initialize_app` from [Firebase Admin SDK](https://firebase.google.com/docs/admin/setup). When `credential_path` is given, it first downloads the credential file and uses it in the `creds` argument of the `initialize_app()`. When `options` parameter of this component is not `None`, it will be passed to the `options` argument of the `initialize_app()`.
  
  - Download the model from the upstream TFX component if the model is blessed. Unfortunately, Firebase ML only lets us upload/host a model from the local storage, so this step is required. Along the way, if the model is `TFLite` format, local flag `is_tfile` will be marked as `True`
  
  - The model format can be either of `SavedModel` or `TFLite`. It searches for the `*.tflite` file during the downloading/copying process. When it is found, `TFLiteGCSModelSource.from_tflite_model_file(model_path)` function will be used to store the model into the GCS bucket specified in `storage_bucket`. If any `*.tflite` is not found, `TFLiteGCSModelSource.from_saved_model(model_path)` is used instead. `from_saved_model()` function internally converts `SavedModel` to `TFLite`, then the rest of the process is the same as `from_tflite_model_file`.

  - Search the list of models whose name is same to the `display_name`. If the list is empty, a new model will be created and hosted. If the list is non-empty, the existing modell will be updated. 
    - In any cases, tags will be updated with the `tags`. Plus, additional tag information of the model version will be automatically added.

## Project Dependencies
The implementation will use the following libraries.
- [Firebase Admin Python SDK](https://github.com/firebase/firebase-admin-python)

## Project Team
**Project Leader** : Chansung Park, deep-diver, deep.diver.csp@gmail.com
1. Sayak Paul, sayakpaul, spsayakpaul@gmail.com
