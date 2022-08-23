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
Firebase ML Publisher component will be implemented as Python class-based component. You can find the [actual source code](https://github.com/deep-diver/complete-mlops-system-workflow/tree/feat/firebase-publisher/training_pipeline/pipeline/components/pusher/FirebasePublisher) in my personal project. 

The implementation details
- This component behaves similar to [Pusher](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Pusher) component, but it pushes/hosts model to Firebase ML instead. To this end, FirebasePublisher interits Pusher, and it gets the following inputs

```python
FirebasePublisher(
  model: types.BaseChannel = None,
  model_blessing: Optional[types.BaseChannel] = None,
  custom_config: Optional[Dict[str, Any]] = None,
)
```

- Each inputs:
  - `model` : the model from the upstream TFX component such as [Trainer](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Trainer)
  - `model_blessing` : the output of `blessing` from the Evaluator component to indicate if the given `model` is good enough to be pushed
  - `custom_config` : additional information to initialize and configure [Firebase Admin SDK](https://firebase.google.com/docs/reference/admin/python). `FIREBASE_ML_MODEL_NAME` and `FIREBASE_ML_MODEL_TAGS` correspond to the `display_name` and `tags` respectively of the Firebase hosted model. `FIREBASE_CREDENTIALS` is a optional parameter, and it indicates GCS location where a Service Account Key (JSON) file is stored. If this parameter is not given, [Application Default Credentials](https://cloud.google.com/docs/authentication/production) will be used in GCP environment

    ```python
    FIREBASE_ML_ARGS = {
        "FIREBASE_ML": {
            "FIREBASE_CREDENTIALS": ...,
            "FIREBASE_ML_MODEL_NAME": ...,
            "FIREBASE_ML_MODEL_TAGS": ["tag1", ... ],
            "OPTIONS": { 
                # to be passed directly into the options argument of firebase_admin.initialize_app. 
                # The mandatory option is storageBucket where Firebase ML stores model
                # https://firebase.google.com/docs/reference/admin/python/firebase_admin#initialize_app

                "storageBucket": ...,
            },
        }
    }
    ```

- It outputs the following information by the method [`_MarkPushed`](https://github.com/tensorflow/tfx/blob/3b5290aa77c2df52a4791715cfd761be7696fe81/tfx/components/pusher/executor.py#L222) from Pusher component
  - `pushed` : indicator if the model is pushed without any issue or if the model is blessed.
  - `pushed_destination` : URL string for easy access to the model from Firebase Console such as `f"https://console.firebase.google.com/u/1/project/{PROJECT_ID}/ml/custom"`
  - `pushed_version` : version string of the pushed model. This is determined in the same manner as Pusher by `str(int(time.time()))`

- The detailed behaviour of this component
  - Initialize Firebase App with `firebase_admin.initialize_app` from [Firebase Admin SDK](https://firebase.google.com/docs/admin/setup). It uses the inputs passed in `custom_config`. When `FIREBASE_CREDENTIALS` is given, it first downloads the credential file. Otherwise, all the values in `OPTIONS` will be passed to the `options` parameter
  
  - Download the model from the upstream TFX component if the model is blessed. Unfortunately, Firebase ML only lets us upload/host a model from the local storage, so this step is required. Along the way, if the model is `TFLite` format, local flag `is_tfile` will be marked as `True`
  
  - If the model is `SavedModel` (which can be determined if `is_tflite` is set to `False`), `TFLiteGCSModelSource.from_saved_model(model_path)` is called. This function will convert `SavedModel` to `TFLite` and stores it in the GCS bucket specified in `storageBucket` of `custom_config`. Otherwise, `TFLiteGCSModelSource.from_tflite_model_file(model_path)` is used to directly upload the given `TFLite` model file

  - Search the list of models whose `display_name` is same to the `FIREBASE_ML_MODEL_NAME` in `custom_config`. If the list is empty, a new model will be created and hosted. If the list is non-empty, the existing modell will be updated. 
    - In any cases, tags will be updated with the `FIREBASE_ML_MODEL_TAGS` in `custom_config`. Plus, additional tag information of the model version will be automatically added.

## Project Dependencies
The implementation will use the following libraries.
- [Firebase Admin Python SDK](https://github.com/firebase/firebase-admin-python)

## Project Team
**Project Leader** : Chansung Park, deep-diver, deep.diver.csp@gmail.com
1. Sayak Paul, sayakpaul, spsayakpaul@gmail.com
