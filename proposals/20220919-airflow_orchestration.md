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
1. tfx/dsl/components/base:base_node
2. tfx/dsl/components/base:base_component
3. tfx/dsl/components/base:base_executor
4. tfx/dsl/components/base:executor_spec
5. tfx/orchestration:data_types
6. tfx/orchestration:metadata
7. tfx/orchestration/config:base_component_config
8. tfx/orchestration/launcher:base_component_launcher
9. tfx/orchestration:pipeline
10. tfx/orchestration:tfx_runner
11. tfx/orchestration/config:config_utils
12. tfx/orchestration/config:pipeline_config
13. tfx/utils:json_utils
14. tfx/utils:telemetry_utils

## Project Team
Varun Murthy (murthyvs@google.com)
Google Core ML TFX Team (tfx-team@google.com)

# Note
Please be aware of the processes and requirements which are outlined here:

* [SIG-TFX-Addons](https://github.com/tensorflow/tfx-addons)
* [Contributing Guidelines](https://github.com/tensorflow/tfx-addons/blob/main/CONTRIBUTING.md)
* [TensorFlow Code of Conduct](https://github.com/tensorflow/tfx-addons/blob/main/CODE_OF_CONDUCT.md)
