#### SIG TFX-Addons
# Project Proposal

**Your name:** Varun Murthy

**Your email:** murthyvs@google.com

**Your company/organization:** Google Core ML TFX Team (MTV + Seoul)

**Project name:** Airflow Orchestration

## Project Description
Moving Airflow to TFX add-ons due to decreasing native support.

## Project Category
Orchestrator

## Project Use-Case(s)
We are moving Airflow from tfx/orchestration to tfx-addons. Native support for Airflow won't be provided in the near future.

## Project Implementation
1. Copy tfx/orchestration/airflow to tfx-addons/airflow.
2. Mark Airflow as deprecated in tfx/ and indicate that support will be dropped in 1-2 releases.
3. Update TFX tutorials on www.tensorflow.org to indicate deprecated and moving to TFXA
 
## Project Dependencies
TensorFlow Extended (1.9.1)

## Project Team
murthyvs: Varun Murthy (murthyvs@google.com)

# Note
Please be aware of the processes and requirements which are outlined here:

* [SIG-TFX-Addons](https://github.com/tensorflow/tfx-addons)
* [Contributing Guidelines](https://github.com/tensorflow/tfx-addons/blob/main/CONTRIBUTING.md)
* [TensorFlow Code of Conduct](https://github.com/tensorflow/tfx-addons/blob/main/CODE_OF_CONDUCT.md)
