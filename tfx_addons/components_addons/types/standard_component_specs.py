from tfx.proto import example_gen_pb2

from tfx.proto import range_config_pb2

from tfx.types import standard_artifacts, standard_component_specs
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter

REMOTE_ZIP_FILE_URI_KEY = "zip_file_uri"


class RemoteZipFileBasedExampleGenSpec(ComponentSpec):
    """File-based ExampleGen component spec."""

    PARAMETERS = {
        standard_component_specs.INPUT_BASE_KEY:
            ExecutionParameter(type=str),
        REMOTE_ZIP_FILE_URI_KEY:
            ExecutionParameter(type=str),
        standard_component_specs.INPUT_CONFIG_KEY:
            ExecutionParameter(type=example_gen_pb2.Input),
        standard_component_specs.OUTPUT_CONFIG_KEY:
            ExecutionParameter(type=example_gen_pb2.Output),
        standard_component_specs.OUTPUT_DATA_FORMAT_KEY:
            ExecutionParameter(type=int),  # example_gen_pb2.PayloadFormat enum.
        standard_component_specs.OUTPUT_FILE_FORMAT_KEY:
            ExecutionParameter(type=int),  # example_gen_pb2.FileFormat enum.
        standard_component_specs.CUSTOM_CONFIG_KEY:
            ExecutionParameter(type=example_gen_pb2.CustomConfig, optional=True),
        standard_component_specs.RANGE_CONFIG_KEY:
            ExecutionParameter(type=range_config_pb2.RangeConfig, optional=True),
    }
    INPUTS = {}
    OUTPUTS = {
        standard_component_specs.EXAMPLES_KEY: ChannelParameter(type=standard_artifacts.Examples),
    }
