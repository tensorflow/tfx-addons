#### SIG TFX-Addons
# Project Proposal

**Your name:** Michal Brys

**Your email:** michal.brys@openx.com

**Your company/organization:** OpenX

**Project name:** Model load test TFX Component

## Project Description
A TFX component performs a load test of the exported TensorFlow model because the prediction time may vary on the model type and structure.

## Project Category
Component

## Project Use-Case(s)
This component could extend a TFX InfraValidator to validate the TensorFlow model serving performance using the load test.
The load test results will tell us:
* if the model will meet the prediction time requirements (essential in large scale implementations),
* if the changes in the new model will make it faster/slower comparing to the baseline model.

## Project Implementation
The load test component will integrate with TFX using the following approach:
* The `InfraValidator TFX Pipeline Component` component creates a TensorFlow Serving endpoint.
* The load test component spins up a pod on the Kubernetes cluster to perform the load test (send a specified volume of the prediction requests to the TensorFlow Serving endpoint).
* The model serving performance metrics (i.e. average prediction time) and the load test specification (the traffic volume, the network protocol used, etc.) are recorded and stored as an ML Metadata artifact.
* Produced ML Metadata artifacts could validate prediction speed compared to the baseline model. Based on that, we can decide if we want to proceed or break the TFX pipeline execution.

### Possible limitations:
* The component will use the model serving endpoint from `InfraValidator TFX Pipeline Component`, it will support only the TensorFlow SavedModel model format, KubeflowDagRunner inside Kubeflow Pipelines and Kubernetes cluster.
* The prediction time is also related to the infrastructure used for the load test. It means that we should only analyze the model prediction time relatively. The absolute numbers could be hard to reproduce in the production environment.

## Project Dependencies
- An open source tool for load testing (i.e. (HTTP) vegeta https://github.com/tsenart/vegeta or (gRPC) ghz https://github.com/bojand/ghz)
- A k8s cluster (since it relies on the `InfraValidator TFX Pipeline Component` which runs only on Kubeflow)
- `InfraValidator TFX Pipeline Component`

## Project Team
Michal Brys, michal.brys@openx.com, [@michalbrys](https://github.com/michalbrys)
  
Hannes Hapke, [@hanneshapke](https://github.com/hanneshapke)
