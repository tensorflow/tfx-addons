import tfx.v1 as tfx
from tfx.dsl.component.experimental.decorators import component



# our custom artifact will be for Feature scores
"""Custom Artifact type"""

class SynthesizedGraph(tfx.types.artifact.Artifact):
  """Output artifact of the SynthesizeGraph component"""
  TYPE_NAME = 'SynthesizedGraphPath'
  PROPERTIES = {
      'span': standard_artifacts.SPAN_PROPERTY,
      'split_names': standard_artifacts.SPLIT_NAMES_PROPERTY,
  }

# our custom component will be feature selection

@component
def MyTrainerComponent(
    training_data: tfx.dsl.components.InputArtifact[tfx.types.standard_artifacts.Examples],
    model: tfx.dsl.components.OutputArtifact[tfx.types.standard_artifacts.Model],
    dropout_hyperparameter: float,
    num_iterations: tfx.dsl.components.Parameter[int] = 10
    ) -> tfx.v1.dsl.components.OutputDict(loss=float, accuracy=float):
  '''My simple trainer component.'''

  records = read_examples(training_data.uri)
  model_obj = train_model(records, num_iterations, dropout_hyperparameter)
  model_obj.write_to(model.uri)

  return {
    'loss': model_obj.loss,
    'accuracy': model_obj.accuracy
  }