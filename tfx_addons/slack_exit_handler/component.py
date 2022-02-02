import json

from absl import logging
from kfp.pipeline_spec import pipeline_spec_pb2
from slack import WebClient
from slack.errors import SlackApiError
from tfx import v1 as tfx
from tfx.utils import proto_utils


@tfx.orchestration.experimental.exit_handler
def SlackExitHandlerComponent(
    final_status: tfx.dsl.components.Parameter[str],
    slack_token: tfx.dsl.components.Parameter[str],
    slack_channel_id: tfx.dsl.components.Parameter[str],
    on_failure_only: tfx.dsl.components.Parameter[int] = 0,
    gcp_region: tfx.dsl.components.Parameter[str] = "us-central1",
):
    """
    Exit handler component for TFX pipelines originally developed by Digits Financial, Inc.
    The handler notifies the user of the final pipeline status via Slack.

    Args:
        final_status: The final status of the pipeline.
        slack_token: The slack token.
        slack_channel_id: The slack channel id.
        on_failure_only: Whether to notify only on failure
                         (default is 0, TFX < 1.6 doesn't support the boolean type).
        gcp_region: The GCP region for the Vertex pipeline execution.

    Returns:
        None

    Exceptions:
        SlackApiError: If the slack API call fails.
    """

    # parse the final status
    pipeline_task_status = pipeline_spec_pb2.PipelineTaskFinalStatus()
    proto_utils.json_to_proto(final_status, pipeline_task_status)
    logging.info(f"SlackExitHandlerComponent: {final_status}")

    status = json.loads(final_status)

    # leave the exit handler if pipeline succeeded and on_failure_only is True
    if status["state"] == "SUCCEEDED" and on_failure_only:
        logging.info(
            "SlackExitHandlerComponent: Skipping Slack notification on success."
        )
        return

    job_id = status["pipelineJobResourceName"].split("/")[-1]
    # Generate message
    if status["state"] == "SUCCEEDED":
        message = f":tada: Pipeline job *{job_id}* completed successfully.\n"
    else:
        message = f":scream: Pipeline job *{job_id}* failed."
        message += f"\n>{status['error']['message']}"

    # TODO: Extract job ids, region from the pipelineJobResourceName if we run outside of central1
    message += f"\nhttps://console.cloud.google.com/vertex-ai/locations/{gcp_region}/pipelines/runs/{job_id}"

    client = WebClient(token=slack_token)
    try:
        response = client.chat_postMessage(channel=slack_channel_id, text=message)
        logging.info(f"SlackExitHandlerComponent: Slack response: {response}")
    except SlackApiError as e:
        logging.error(
            "SlackExitHandlerComponent: Slack API call failed: "
            f"{e.response['error']}"
        )
