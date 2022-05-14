# TFX Addons

[![TFX Addons package CI](https://github.com/tensorflow/tfx-addons/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorflow/tfx-addons/actions/workflows/ci.yml)
[![TFX Addons CI for examples](https://github.com/tensorflow/tfx-addons/actions/workflows/ci_examples.yml/badge.svg)](https://github.com/tensorflow/tfx-addons/actions/workflows/ci_examples.yml)
[![PyPI](https://badge.fury.io/py/tfx-addons.svg)](https://badge.fury.io/py/tfx-addons)


SIG TFX-Addons is a community-led open source project. As such, the project depends on public contributions, bug fixes, and documentation. This project adheres to the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Maintainership

The maintainers of TensorFlow Addons can be found in the [CODEOWNERS](CODEOWNERS) file of the repo. If you would
like to maintain something, please feel free to submit a PR. We encourage multiple 
owners for all submodules.


## Installation

TFX Addons is available on PyPI for all OS. To install the latest version, 
run the following:

```
pip install tfx-addons
```

To ensure you have a compatible version of dependencies for any given project, 
you can specify the project name  as an extra requirement during install:

```
pip install tfx-addons[feast_examplegen,schema_curation]
``` 

To use TFX Addons:

```python
from tfx import v1 as tfx
import tfx_addons as tfxa

# Then you can easily load projects tfxa.{project_name}. Ex:

tfxa.feast_examplegen.FeastExampleGen(...)

```


## TFX Addons projects

* [tfxa.mlmd_client](tfx_addons/mlmd_client) 
* [tfxa.schema_curation](tfx_addons/schema_curation) 
* [tfxa.feast_examplegen](tfx_addons/feast_examplegen) 
* [tfxa.xgboost_evaluator](tfx_addons/xgboost_evaluator)
* [tfxa.sampling](tfx_addons/sampling)
* [tfxa.message_exit_handler](tfx_addons/message_exit_handler) 


Check out [proposal](proposals) for a list of existing or upcoming projects proposal for TFX Addons


## Tutorials and examples
See [`examples/`](examples/)
for end-to-end examples of various addons.

## Contributing

TFX Addons is a community-led project. Please have a look at our contributing and development guides if you want to contribute to the project: [CONTRIBUTING.md](CONTRIBUTING.md)

### Meeting cadence:

We meet bi-weekly on Wednesday. Check out our [Meeting notes](https://docs.google.com/document/d/1T0uZPoZhwNStuKkeCNsfE-kfc-PINISKIitYxkTK3Gw/edit?resourcekey=0-N9vT9Tn171wYplyYn4IPjQ) and join [sig-tfx-addons@tensorflow.com](https://groups.google.com/a/tensorflow.org/g/sig-tfx-addons) to get invited to the meeting.

## Package releases

Check out [RELEASE.md](RELEASE.md) to learn how TFX Addons is released.

## Resources

- [sig-tfx-addons@tensorflow.org](https://groups.google.com/a/tensorflow.org/g/sig-tfx-addons) – Join our Google group
- [tfx@tensorflow.org](https://groups.google.com/a/tensorflow.org/g/tfx) – General TFX mailing list
- [TFX Addons Slack](https://tfxaddons.slack.com) -  join [here](https://join.slack.com/t/tfxaddons/shared_invite/zt-tu1981lj-npIhRSHF8gl9G0ldUovbcw)
- [SIG Repository](http://github.com/tensorflow/tfx-addons) (this repo)
- [SIG Charter](https://github.com/tensorflow/community/blob/master/sigs/tfx-addons/CHARTER.md)

