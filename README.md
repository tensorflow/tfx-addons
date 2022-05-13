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

## Maintainership

TFX Addons has been designed to compartmentalize submodules so 
that they can be maintained by community users who have expertise, and a vested 
interest in that component. We heavily encourage users to submit sign up to maintain a 
submodule by submitting your username to the [CODEOWNERS](CODEOWNERS) file.

Full write access will only be granted after substantial contribution 
has been made in order to limit the number of users with write permission. 
Contributions can come in the form of issue closings, bug fixes, documentation, 
new code, or optimizing existing code. Submodule maintainership can be granted 
with a lower barrier for entry as this will not include write permissions to 
the repo.


### SIG Membership

We encourage any developers working in production ML environments, infrastructure, or applications to [join and participate in the activities of the SIG](http://goo.gle/tfx-addons-group). Whether you are working on advancing the platform, prototyping or building specific applications, or authoring new components, templates, libraries, and/or orchestrator support, we welcome your feedback on and contributions to TFX and its tooling, and are eager to hear about any downstream results, implementations, and extensions.

We have multiple channels for participation, and publicly archive discussions in our user group mailing list:
- tfx-addons@tensorflow.org – Google group for SIG TFX-Addons
- tfx@tensorflow.org – our general mailing list for TFX
- [TFX Addons Slack](https://tfxaddons.slack.com) -  Our shared slack workspace (join [here](https://join.slack.com/t/tfxaddons/shared_invite/zt-tu1981lj-npIhRSHF8gl9G0ldUovbcw) )

Other Resources
- SIG Repository: http://github.com/tensorflow/tfx-addons (this repo)
- Documentation: https://www.tensorflow.org/tfx
- SIG Charter:  https://github.com/tensorflow/community/blob/master/sigs/tfx-addons/CHARTER.md

Meeting cadence:
- Bi-weekly on Wednesday. [Meeting notes](https://docs.google.com/document/d/1T0uZPoZhwNStuKkeCNsfE-kfc-PINISKIitYxkTK3Gw/edit?resourcekey=0-N9vT9Tn171wYplyYn4IPjQ)


### Periodic Evaluation of Components and Examples

Components may become less and less useful to the community and TFX examples might become outdated as future TFX versions are released. In order to keep the repository sustainable, we'll be performing bi-annual reviews of our code to ensure everything still belongs within the repo. Contributing factors to this review will be:

1. Number of active maintainers
2. Amount of issues or bugs attributed to the code
3. If a better solution is now available

Functionality within TFX Addons can be categorized into three groups:

* **Suggested**: well-maintained components and examples; use is encouraged.
* **Discouraged**: a better alternative is available; the API is kept for historic reasons; or the components and examples require maintenance and is the waiting period to be deprecated.
* **Deprecated**: use at your own risk; subject to be deleted.

The status change between these three groups is: Suggested <-> Discouraged -> Deprecated.

The period between an API being marked as deprecated and being deleted will be 90 days. The rationale being:
In the event that TFX Addons releases monthly, there will be 2-3 releases before an API is deleted. The release notes could give user enough warning. 90 days gives maintainers ample time to fix their code.

### Project Approvals
1. Project proposals will be submitted to the SIG and published for open review and comment by SIG members for 2 weeks.
2. Following review and approval by the Google TFX team, core team members will vote either in person or offline on whether to approve or reject project proposals.
3. All projects must meet the following criteria:
   - Team members must be named in the proposal
   - All team members must have completed a [Contributor License Agreement](https://cla.developers.google.com/)
   - The project must not violate the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md), [Google AI Principles](https://ai.google/principles/) or [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/).
4. Projects must code to supported open interfaces only, and not reach into core TFX to make changes or rely on private classes, methods, properties, or interfaces.
5. **Google retains the right to reject any proposal.**
6. Projects must first be approved by the Google team.  Projects are then sent for approval to the core community team.  Projects will be approved with a minimum of three `+1` votes, but can be sent for changes and re-review with a single `-1` vote.
