# Project Proposal for Slack Exit Handler for TFX Pipelines

[![Python](https://img.shields.io/pypi/pyversions/tfx.svg?style=plastic)](https://github.com/tensorflow/tfx)
[![TensorFlow](https://img.shields.io/badge/TFX-orange)](https://www.tensorflow.org/tfx)


## Project Description

This component provides an exit handler for TFX pipelines which notifies the user about the final state of the pipeline (failed or succeeded) via a Slack message. If the pipeline failed, the component will provide the error message.


## Project Use-Case(s)

The exit handler notifies about the final state of a pipeline. Instead of constantly pulling the pipeline status via the Vertex cli, the exit handler notifies the users subscripbed to a Slack channel.

The implementation can be extended to cover us communication services (e.g. SMS via Twilio) too.

## Project Implementation

The existing implementation is Python-based and it uses the `tfx.orchestration.experimental.exit_handler` decorator.

The component excepts 4 parameters:
* final_status
* slack_token
* slack_channel_id
* on_failure_only
* gcp_region

`final_status` is the JSON string of the pipeline status, provided by TFX. The Slack parameters contain the credentials to submit the message.  And `on_failure_only` is a configuration for frequently run pipeline to only alert on failures. We have a number of pipelines were this options was useful.

The component parses the status, and composes a message based on the content.

```
    job_id = status["pipelineJobResourceName"].split("/")[-1]
    if status["state"] == "SUCCEEDED":
        message = f":tada: Pipeline job *{job_id}* completed successfully.\n"
    else:
        message = f":scream: Pipeline job *{job_id}* failed."
        message += f"\n>{status['error']['message']}"
```

The a Slack web client object is created and the message is submitted via the object.

Overall, the implementation is minimal, but it serves as a great exit handler example.

### Usage Example at Digits

#### Example usage

```
    from tfx_addons.slack_exit_handler.component import SlackExitHandlerComponent

    ...

    dsl_pipeline = pipeline.create_pipeline(
        ...
    )

    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=runner_config,
    )

    exit_handler = SlackExitHandlerComponent(
        final_status=tfx.orchestration.experimental.FinalStatusStr(),
        slack_token=YOUR_SLACK_TOKEN,
        slack_channel_id=YOUR_SLACK_CHANNEL_ID,
        gcp_region="us-central1",
    )
    runner.set_exit_handler(exit_handler)
    runner.run(pipeline=dsl_pipeline, write_out=True)
```

#### Pipeline Success Message
![Screen_Shot_2022-01-05_at_3_23_43_PM_2](https://user-images.githubusercontent.com/1234819/148304418-9232fe68-57a3-4976-bd01-8d3e14bbf00b.png)

#### Pipeline Failure Message
![_Screen_Shot_2022-01-05_at_2_45_47_PM](https://user-images.githubusercontent.com/1234819/148301546-b8ae19e3-ff71-4ec6-9969-06e71672b2e2.png)

#### Visualization in Google Cloud Vertex Pipelines
![Screen_Shot_2022-01-05_at_3_28_06_PM_2](https://user-images.githubusercontent.com/1234819/148304482-22347d1f-fb9c-4744-92ef-1d020c79f2fc.png)


## Project Dependencies

The component requires:
* TFX version >= 1.4.0
* Slack Python client

The component will also require Google Cloud's Vertex pipelines as its orchestrator.

## Project Team

* Hannes Max Hapke (@hanneshapke), Digits Financial, Inc., hannes -at- digits.com
