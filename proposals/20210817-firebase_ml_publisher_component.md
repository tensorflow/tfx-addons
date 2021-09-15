#### SIG TFX-Addons
# Project Proposal

**Your name:** Chansung Park

**Your email:** deep.diver.csp@gmail.com

**Your company/organization:** Individual(ML GDE)

**Project name:** [Firebase ML Publisher](https://github.com/tensorflow/tfx-addons/issues/59)

## Project Description
This project defines a custom TFX component to publish/update ML models to [Firebase ML](https://firebase.google.com/products/ml). This is another type of pusher component, and the input model is assumed to be a TFLite format.

## Project Category
Component

## Project Use-Case(s)
This project helps users to publish trained models directly to Firebase ML.

With Firebase ML, we can guarantee that mobile devices can be equipped with the latest ML model without explicitly embedding binary in the project compiling stage. We can even A/B test different versions of a model with Google Analytics when the model is published on Firebase ML.

## Project Implementation
Firebase ML Publisher component will be implemented as Python function-based component. You can find the [actual source code](https://github.com/sayakpaul/Dual-Deployments-on-Vertex-AI/blob/main/custom_components/firebase_publisher.py) in my personal project. Please note this is a personal implementation, and it will be enhanced as a official TFX Addon component.

The implementation details
- Define a custom Python function-based TFX component. It takes the following parameters from a previous component.
  - It should follow the standard Pusher's interface since this is another custom pusher.
  - Additionally, it takes meta information to manage published model for Firebase ML such as `display name` and `tags`.
- Download saved TFLite model file by referencing the output from a previous component
  - Firebase SDK doesn't allow to publish models from GCS directly.
- Initialize Firebase Admin with the credential and Firebase temporary-use GCS bucket.
  - Firebase credentials can be setup via [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) for GKE or [Mounting Secret API in TFX runner](https://github.com/tensorflow/tfx/blob/d989bbd7fc366c73ad833428ce6b5cf57a587432/tfx/orchestration/kubeflow/kubeflow_dag_runner.py#L78).
- Search if any models with the same `display name` has already been published.
  - if yes, update the existing Firebase ML mode, then publish it
  - if no, create a new Firebase ML model, then publish it
- Return `tfx.dsl.components.OutputDict` to indicate if the job went successful, and if the job was about creating a new Firebase ML model or updating the exisitng Firebase ML model.

## Project Dependencies
The implementation will use the following libraries.
- [Firebase Admin Python SDK](https://github.com/firebase/firebase-admin-python) >= 5.0.2
- [Python Client for Google Cloud Storage](https://github.com/googleapis/python-storage) >= 1.42.0

## Project Team
**Project Leader** : Chansung Park, deep-diver, deep.diver.csp@gmail.com
1. Sayak Paul, sayakpaul, spsayakpaul@gmail.com
