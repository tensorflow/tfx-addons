#### SIG TFX-Addons
# Project Proposal

**Your name:** Karan Shukla

**Your email:** karanshukla@google.com

**Your company/organization:** Google

**Project name:** Model Card Component

## Project Description
Add support for generating model cards from TFX pipeline configurations.

## Project Category
Component

## Project Use-Case(s)
This component will be used for generating [model cards](https://arxiv.org/abs/1810.03993) for TFX pipelines. This will be a report displaying quantitative information about a model's performance (from TFMA), its data (from TFDV), as well as qualitative information (provided via JSON).

## Project Implementation
The ModelCardComponent is a new fully custom component. It will run after the Pusher component, and produces model cards for blessed models.

The ModelCardComponent will be composed of a ModelCardComponentSpec and ModelCardExecutor. The details of these are described below.

### ModelCardComponentSpec

The ModelCardComponentSpec will accept the following standard artifacts as input:
* ExampleStatistics
* ModelEvaluation
* PushedModel

It will also accept the following optional parameters:
* `model_card_json`, to allow the manual population of free-form text fields such as those in the Considerations section
* `model_card_template`, to provide custom model views

The ModelCardComponentSpec outputs a ModelCardArtifact. The artifact uri points to the directory where the ModelCard assets (proto, html, etc.) are stored. The artifact name is a unique identifier composed of the model's name, version, and date, appended with the time of artifact creation.

### ModelCardExecutor

The ModelCardExecutor will accept the above parameters, inputs, and outputs as args.
It will create a TfxModelCardToolkit instance and run through the standard MCT workflow (see Outline below).

### TfxModelCardToolkit

TfxModelCardToolkit inherits from ModelCardToolkit and follows a similar API, with a few differences to match TFX component APIs.

**Input Artifacts**: While ModelCardToolkit discovers Artifacts by searching a MetadataStore, TfxModelCardToolkit will directly accept Artifacts as inputs instead.

**Output Artifact**: TfxModelCardToolkit will write the model card assets to the uri specified in the ModelCardArtifact. TFX will then handle writing this ModelCardArtifact to MLMD.

The TfxModelCardToolkit will also use the PushedModel's uri to populate a new `ModelCard.model_details.path` field.

## Outline

```python
class ModelCardArtifact(Artifact):
  TYPE_NAME = 'ModelCard'


class ModelCardComponentSpec(component_spec.ComponentSpec):
  PARAMETERS = {
      'model_card_json':
          component_spec.ExecutionParameter(type=Text),
      'model_card_template':
 component_spec.ExecutionParameter(type=Text)
  }
  INPUTS = {
      'example_statistics_artifact':
          component_spec.ChannelParameter(
              type=standard_artifacts.ExampleStatistics),
      'pushed_model_artifact':          component_spec.ChannelParameter(type=standard_artifacts.PushedModel),
      'model_evaluation_artifact':
          component_spec.ChannelParameter(
              type=standard_artifacts.ModelEvaluation),
  }
  OUTPUTS = {
      'model_card_artifact':
          component_spec.ChannelParameter(type=ModelCardArtifact),
  }


class ModelCardExecutor(BaseBeamExecutor):

  def Do(self, input_dict: Dict[str, List[Artifact]],
         output_dict: Dict[str, List[Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    # Initialize ModelCardToolkit with input and output artifacts
    mct = TfxModelCardToolkit(
        example_statistics_artifact=input_dict['dataset'],
        pushed_model_artifact=input_dict['pushed_model_artifact'],
        model_evaluation_artifact=input_dict['model_evaluation_artifact'],
        output_dir=output_dict['model_card_artifact'])
    # Create model card from input artifacts and proto
    model_card = mct.scaffold_assets()
    model_card.from_json(exec_properties['model_card_json'])
    # Write model card as output artifact
    mct.export_format(template_path=exec_properties['model_card_template'])


class TfxModelCardToolkit(ModelCardToolkit):

  def __init__(self,
               example_statistics_artifact: Optional[
                   standard_artifacts.ExampleStatistics] = None,
               pushed_model_artifact: Optional[standard_artifacts.PushedModel] = None,
               model_evaluation_artifact: Optional[
                   standard_artifacts.ModelEvaluation] = None,
               output_dir: Optional[str] = None):
    self._example_statistics_artifact = example_statistics_artifact
    self._pushed_model_artifact = pushed_model_artifact
    self._model_evaluation_artifact = model_evaluation_artifact
    super().__init__(output_dir=output_dir)

  def _scaffold_model_card(self) -> ModelCard:
    # TODO(karanshukla): populate ModelCard using input artifacts
```

### Packaging and Release

This will be part of the existing [`model-card-toolkit`]((https://pypi.org/project/model-card-toolkit/)) package, which will be handled and released by the project team. The code will live in the existing [model-card-toolkit](https://github.com/tensorflow/model-card-toolkit) repository, under the new path `model_card_toolkit/tfx`.

## Project Dependencies
The model-card-toolkit [dependencies](https://github.com/tensorflow/model-card-toolkit/blob/master/setup.py) include [jinja2](https://pypi.org/project/Jinja2/), [matplotlib](https://pypi.org/project/matplotlib/), [tensorflow-model-analysis](https://pypi.org/project/tensorflow-model-analysis/), and [ml-metadata](https://pypi.org/project/ml-metadata/).

## Project Team
Karan Shukla, shuklak13, karanshukla@google.com

# Note
Please be aware of the processes and requirements which are outlined here:

* [SIG-TFX-Addons](https://github.com/tensorflow/tfx-addons)
* [Contributing Guidelines](https://github.com/tensorflow/tfx-addons/blob/main/CONTRIBUTING.md)
* [TensorFlow Code of Conduct](https://github.com/tensorflow/tfx-addons/blob/main/CODE_OF_CONDUCT.md)
