#### SIG TFX-Addons
# Project Proposal

**Your name:** Chansung Park

**Your email:** deep.diver.csp@gmail.com

**Your company/organization:** Individual(ML GDE)

**Project name:** [Firebase ML Publisher](https://github.com/tensorflow/tfx-addons/issues/59)

## Project Description
This project defines a custom TFX component to publish/update ML models from TFX Pusher to [Firebase ML](https://firebase.google.com/products/ml). The input model from TFX Pusher is assumed to be a TFLite format.

## Project Category
Component

## Project Use-Case(s)
This project helps users to publish trained models directly from TFX Pusher component to Firebase ML. 

With Firebase ML, we can guarantee that mobile devices can be equipped with the latest ML model without explicitly embedding binary in the project compiling stage. We can even A/B test different versions of a model with Google Analytics when the model is published on Firebase ML.

## Project Implementation
Firebase ML Publisher component will be implemented as Python function-based component. You can find the [actual source code](https://github.com/sayakpaul/Dual-Deployments-on-Vertex-AI/blob/main/custom_components/firebase_publisher.py) in my personal project. 

The implementation details
- Define a custom Python function-based TFX component. It takes the following parameters from a previous component.
  - The URI of the pushed model from TFX Pusher component.
  - Requirements from Firebase ML (credential JSON file path, Firebase temporary-use GCS bucket). Please find more information from [Before you begin section](https://firebase.google.com/docs/ml/manage-hosted-models#before_you_begin) in the official Firebase document.
  - Meta information to manage published model for Firebase ML such as `display name` and `tags`.
- Download the Firebase credential file and pushed TFLite model file.
- Initialize Firebase Admin with the credential and Firebase temporary-use GCS bucket.
- Search if any models with the same `display name` has already been published.
  - if yes, update the existing Firebase ML mode, then publish it
  - if no, create a new Firebase ML model, then publish it
- Return `tfx.dsl.components.OutputDict(result=str)` to indicate if the job went successful, and if the job was about creating a new Firebase ML model or updating the exisitng Firebase ML model.

## Project Dependencies
The implementation will use the following libraries.
- [Firebase Admin Python SDK](https://github.com/firebase/firebase-admin-python)
- [Python Client for Google Cloud Storage](https://github.com/googleapis/python-storage) 

## Project Team
**Project Leader** : Chansung Park, deep-diver, deep.diver.csp@gmail.com
1. Sayak Paul, sayakpaul, spsayakpaul@gmail.com