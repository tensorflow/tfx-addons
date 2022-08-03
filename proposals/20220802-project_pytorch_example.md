**Your name:** Hannes Hapke

**Your email:** hannes--at--digits--dot--com

**Your company/organization:** Digits Financial Inc

**Project name:** TFX PyTorch Example

## Project Description
Adding a TFX pipeline example for PyTorch models to the TFX Addons repository.

## Project Category
Example

## Project Use-Case(s)
While there are a few non-TF model-based examples for TFX (e.g. JAX or Scikit), there isn't a maintained example for PyTorch models.

## Project Implementation
The pipeline example includes the following components:
- Load a known dataset, e.g. MNIST,  via the CSVExampleGen component
- Run the standard statistics and schema steps via StatisticsGen and SchemaGen
- Performs a pseudo transformation (passthrough of the values) with the new PandasTransform component from tfx-addons
- Add a custom run_fn function for PyTorch for the Trainer component
- Add a TFMA example how to analysis PyTorch models to obtain a model blessing
- Push the models to a local path

## Project Dependencies
The example will on TFX (1.9.1), TFX addons (0.2), Apache Beam, and PyTorch (1.0.2)

## Project Team
Hannes Hapke (gh: hanneshapke, email: hannes--at--digits--dot--com)
More contributors more than welcome

# Note
Please be aware of the processes and requirements which are outlined here:

* [SIG-TFX-Addons](https://github.com/tensorflow/tfx-addons)
* [Contributing Guidelines](https://github.com/tensorflow/tfx-addons/blob/main/CONTRIBUTING.md)
* [TensorFlow Code of Conduct](https://github.com/tensorflow/tfx-addons/blob/main/CODE_OF_CONDUCT.md)
