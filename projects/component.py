from tfx import v1 as tfx
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.components.util import tfxio_utils
from tfx.dsl.component.experimental.decorators import component
import apache_beam as beam
import random

@component
def UndersamplingComponent(
    examples: tfx.dsl.components.InputArtifact[tfx.types.standard_artifacts.Examples]
    ) -> tfx.dsl.components.OutputArtifact[tfx.types.standard_artifacts.Examples]:
        # examples = artifact_utils.get_single_instance(examples.outputs["examples"]._artifacts)
        tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(examples=[examples], telemetry_descriptors=[])
        split_and_tfxio = [(split, tfxio_factory(io_utils.all_files_pattern(artifact_utils.get_split_uri([examples], split))))
                        for split in artifact_utils.decode_split_names(examples.split_names)]

        def generate_elements(data):
            for i in range(len(data[list(data.keys())[0]])):
                yield {key: data[key][i][0] if len(data[key][i]) > 0 else "" for key in data.keys()}

        def sample(key, value, side=0):
            for item in random.sample(value, side):
                yield item
            
        split, tfxio = split_and_tfxio[0]
        # TODO: alter this code to account for multiple files in the uri
        uri = examples.uri.split("/")
        uri[3] = "UndersamplingComponent"
        uri.append(f"Split-{split}")
        uri = "/".join(uri)

        with beam.Pipeline() as p:
            data = (
                # TODO: convert to list and back using a schema to save key space?
                p 
                | 'TFXIORead[%s]' % split >> tfxio.BeamSource()
                | beam.Map(lambda x: x.to_pydict())
                | beam.FlatMap(generate_elements)
                | beam.Map(lambda x: (x["company"], x))
            )
            
            val = (
                data
                | beam.combiners.Count.PerKey()
                | beam.Values()
                | beam.CombineGlobally(lambda elements: min(elements or [-1]))
            )

            res = (
                data 
                | beam.GroupByKey()
                | beam.FlatMapTuple(sample, side=beam.pvalue.AsSingleton(val))
                | beam.io.WriteToTFRecord(uri, file_name_suffix='.gz')
            )

