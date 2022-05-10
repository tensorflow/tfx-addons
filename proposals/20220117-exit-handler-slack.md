#### SIG TFX-Addons
# Project Proposal for Slack Exit Handler for TFX Pipelines

**Your name:** Hannes Max Hapke

**Your email:** hannes@digits.com

**Your company/organization:** Digits Financial, Inc.

**Project name:** Slack Exit Handler for TFX Pipelines

## Project Description

The component provides an exit handler for TFX pipelines which notifies the user about the final state of the pipeline (failed or succeeded) via a Slack message. If the pipeline failed, the component will provide the error message.

## Project Category

Component

## Project Use-Case(s)

The exit handler notifies Digits' ML team about the final state of a pipeline. Instead of constantly pulling the pipeline status via the Vertex cli, the exit handler notifies us.

The implementation can be extended to cover us communication services (e.g. SMS via Twilio) too.

Furthermore, the implementation can be seen as an example implementation for an exit handler. Other users could use the same setup to trigger downstream pipelines or trigger other post-run actions.

## Project Implementation

The existing implementation is Python-based and it uses the `tfx.orchestration.experimental.exit_handler` decorator.

The component excepts 4 parameters:
* final_status
* slack_token
* slack_channel_id
* on_failure_only

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

### Current Digits Implementation

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

* Hannes Hapke (@hanneshapke), hannes -at- digits.com

# Note

Please be aware of the processes and requirements which are outlined here:

* [SIG-TFX-Addons](https://github.com/tensorflow/tfx-addons)
* [Contributing Guidelines](https://github.com/tensorflow/tfx-addons/blob/main/CONTRIBUTING.md)
* [TensorFlow Code of Conduct](https://github.com/tensorflow/tfx-addons/blob/main/CODE_OF_CONDUCT.md)
