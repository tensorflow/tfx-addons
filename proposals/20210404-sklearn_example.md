#### SIG TFX-Addons

# Project Proposal

**Your name:** Michael Hu

**Your email:** humichael@google.com

**Your company/organization:** Google

**Project name:** Scikit-learn Penguin Classification 

## Project Description
Demonstrates training a scikit-learn MLPClassifier model in a TFX pipeline. The pipeline can either run locally or on GCP using CAIP, Dataflow, and Kubeflow Pipelines.

## Project Category
Example

## Project Use-Case(s)
This example can be used to push any scikit-learn model to CAIP with minimal custom code to acquire standard TFX benefits like orchestration, data validation, gated retraining, etc. This example is currently not used within my organization.

## Project Implementation
Scikit-learn will be integrated with TFX by using the following approach:
Create a custom trainer module for training the scikit-learn model using example protos.

Tensors parsed from examples will be converted to Numpy arrays.
The model artifact will be stored as a pickle, which both the custom evaluator module and CAIP serving will be able to load.

Create a custom evaluator module for making predictions against the model in Evaluator.

Build a Docker container extending a TFX image for managing the scikit-learn version and dependencies when training on CAIP. This container will be hosted in the user's Google Container Registry on GCP.

CAIP supports serving scikit-learn models out of the box.

The project will not be packaged. Instead, users just need to clone the source code to run the example.

## Project Dependencies
'scikit-learn>=0.23,<0.24'
kfp

## Project Team
Michael Hu, humichael@google.com

Jiayi Zhao, jyzhao@google.com 

