from typing import Optional, Union

from tfx import types
from tfx.components.example_gen import driver
from tfx.components.example_gen import utils
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2
from tfx.types import standard_artifacts
from census_consumer_complaint_types.types import RemoteZipFileBasedExampleGenSpec


class RemoteZipFileBasedExampleGen(base_beam_component.BaseBeamComponent):
    """A TFX component to ingest examples from a file system.

  The RemoteZipFileBasedExampleGen component is an API for getting zip
  file-based available at HTTP urlrecords into TFX pipelines. It consumes
  external files to generate examples which will
  be used by other internal components like StatisticsGen or Trainers.  The
  component will also convert the input data into
  [tf.record](https://www.tensorflow.org/tutorials/load_data/tf_records)
  and generate train and eval example splits for downstream components.

  ## Example
  ```
  _taxi_root = os.path.join(os.environ['HOME'], 'taxi')
  _data_root = os.path.join(_taxi_root, 'data', 'simple')
  _zip_uri = "https://xyz//abz.csv.zip"
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = RemoteZipFileBasedExample(input_base=_data_root,zip_file_uri="")
  ```

  Component `outputs` contains:
   - `examples`: Channel of type `standard_artifacts.Examples` for output train
                 and eval examples.
  """

    SPEC_CLASS = RemoteZipFileBasedExampleGenSpec

    # EXECUTOR_SPEC should be overridden by subclasses.
    EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(
        base_beam_executor.BaseBeamExecutor)
    DRIVER_CLASS = driver.FileBasedDriver

    def __init__(
            self,
            input_base: Optional[str] = None,
            zip_file_uri: Optional[str] = None,
            input_config: Optional[Union[example_gen_pb2.Input,
                                         data_types.RuntimeParameter]] = None,
            output_config: Optional[Union[example_gen_pb2.Output,
                                          data_types.RuntimeParameter]] = None,
            custom_config: Optional[Union[example_gen_pb2.CustomConfig,
                                          data_types.RuntimeParameter]] = None,
            range_config: Optional[Union[range_config_pb2.RangeConfig,
                                         data_types.RuntimeParameter]] = None,
            output_data_format: Optional[int] = example_gen_pb2.FORMAT_TF_EXAMPLE,
            output_file_format: Optional[int] = example_gen_pb2.FORMAT_TFRECORDS_GZIP,
            custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None):
        """Construct a FileBasedExampleGen component.

    Args:
      input_base: an extract directory containing the CSV files after extraction of downloaded zip file.
      zip_file_uri: Remote Zip file uri to download compressed zip csv file
      input_config: An
        [`example_gen_pb2.Input`](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)
          instance, providing input configuration. If unset, input files will be
          treated as a single split.
      output_config: An example_gen_pb2.Output instance, providing the output
        configuration. If unset, default splits will be 'train' and
        'eval' with size 2:1.
      custom_config: An optional example_gen_pb2.CustomConfig instance,
        providing custom configuration for executor.
      range_config: An optional range_config_pb2.RangeConfig instance,
        specifying the range of span values to consider. If unset, driver will
        default to searching for latest span with no restrictions.
      output_data_format: Payload format of generated data in output artifact,
        one of example_gen_pb2.PayloadFormat enum.
      output_file_format: File format of generated data in output artifact,
        one of example_gen_pb2.FileFormat enum.
      custom_executor_spec: Optional custom executor spec overriding the default
        executor spec specified in the component attribute.
    """
        # Configure inputs and outputs.
        input_config = input_config or utils.make_default_input_config()
        output_config = output_config or utils.make_default_output_config(
            input_config)
        example_artifacts = types.Channel(type=standard_artifacts.Examples)
        spec = RemoteZipFileBasedExampleGenSpec(
            input_base=input_base,
            zip_file_uri=zip_file_uri,
            input_config=input_config,
            output_config=output_config,
            custom_config=custom_config,
            range_config=range_config,
            output_data_format=output_data_format,
            output_file_format=output_file_format,
            examples=example_artifacts)
        super().__init__(spec=spec, custom_executor_spec=custom_executor_spec)
