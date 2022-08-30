# Contributing guidelines

Interested in contributing to TFX Addons? We appreciate all kinds of help and are working to make this guide as comprehensive as possible.
Please let us know if you think of something we could do to help lower the barrier to contributing.

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement (CLA).

  * If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work, then you'll need to sign a [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository.

### SIG Membership

We encourage any developers working in production ML environments, infrastructure, or applications to [join and participate in the activities of the SIG](http://goo.gle/tfx-addons-group). Whether you are working on advancing the platform, prototyping or building specific applications, or authoring new components, templates, libraries, and/or orchestrator support, we welcome your feedback on and contributions to TFX and its tooling, and are eager to hear about any downstream results, implementations, and extensions.

### Project Maintainership 

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


### Project proposals

If you have a new project, you can contribute it to TFX Addons! Before doing so, make sure to submit a project proposal to 
the repository under [proposals/](proposals/). Use [proposals/yyyymmdd-project_template.md](proposals/yyyymmdd-project_template.md) as a template to get started.

1. Project proposals will be submitted to the SIG and published for open review and comment by SIG members for 2 weeks.
2. Following review and approval by the Google TFX team, core team members will vote either in person or offline on whether to approve or reject project proposals.
3. All projects must meet the following criteria:
   - Team members must be named in the proposal
   - All team members must have completed a [Contributor License Agreement](https://cla.developers.google.com/)
   - The project must not violate the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md), [Google AI Principles](https://ai.google/principles/) or [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/).
4. Projects must code to supported open interfaces only, and not reach into core TFX to make changes or rely on private classes, methods, properties, or interfaces.
5. **Google retains the right to reject any proposal.**
6. Projects must first be approved by the Google team.  Projects are then sent for approval to the core community team.  Projects will be approved with a minimum of three `+1` votes, but can be sent for changes and re-review with a single `-1` vote.



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

## Contributing code

If you have improvements to TFX Addons, send us your pull requests! For those
just getting started, Github has a [howto](https://help.github.com/articles/using-pull-requests/).

SIG team members will be assigned to review your pull requests. Once the pull requests are approved and pass continuous integration checks, we will merge the pull requests.

### Reviewing contributions

- Reviewers can approve pull requests using `/lgtm` command. CODEOWNERS file is used to validate approvals.
- Reviewers can request an automatic merge using `/merge` command. Auto-merge will be performed after all tests have passed and enough approvals are recollected for the PR.
- Changes to `.github/workflows` will need to be merged manually by a reviewer with write permissions.

### Code Ownership

* Code ownership is tracked through the `CODEOWNERS` file. Users can be added if one of the following situations apply:
  * When a project proposal is approved, the initial contributors become automatically code owner of the project folder.
  * Developers who contribute or maintain a TFX Addons component, example, etc. can gain coder owner access to the project folders
    if the initial contributors agree.
  * In case, the initial contributors have abandoned the project or can't be reached, the TFX Addons core team can decide about the ownership reassignment.
  * Requesting project code ownership requires a substantial contribution (e.g. update of a component to a newer TFX version).

### Specifying project dependencies

Each project specifies it's own Python dependencies depending on what folder it lives under:

* **Projects in `examples/`**: Those need to provide a `requirements.txt` in the root of their folder. Example: `examples/xgboost_penguins/requirements.txt`. You can depend on a `tfx_addons` project by using `../..[project_name]` in your `requirements.txt` file.
* **Projects in `tfx_addons/`**: In order for project to be included in release and be tested, you will need to specify dependencies in [tfx_addons/version.py](https://github.com/tensorflow/tfx-addons/blob/main/tfx_addons/version.py) `_PKG_METADATA` where key is the project name (aka tfx_addons/{project_name}) and value is a list of requirements strings needed for your component. Once added, this will automatically be picked up by CI and will automatically include your project into the tfx-addons release. In addition, your project will be added to the `tfx_addons.{project_name}` namespace, such that it can be used:

```python

import tfx_addons as tfxa

tfxa.project_name
```

Note that CI runs on `pytest`, see _Testing your code_ below to check how to create tests for your code.

### Development tips

We use [pre-commit](https://pre-commit.com/) to validate our code before we push to the repository. We use push over commit to allow more flexibility.

Here's how to install it locally:
- Create virtual environemnt: `python3 -m venv env`
- Activate virtual environment: `source env/bin/activate`
- Upgrade pip: `pip install --upgrade pip`
- Install test packages: `pip install -e ".[test]"`
- Install pre-commit hooks for push hooks: `pre-commit install --hook-type pre-push`
- Change and commit files. pre-commit will run the tests and linting before you push. You can also manually trigger the tests and linting with `pre-commit run --hook-stage push --all-files`

Note that pre-commit will be run via GitHub Action automatically for new PRs.

### Contribution guidelines and standards

Before sending your pull request for
[review](https://github.com/tensorflow/tfx-addons/pulls),
make sure your changes are consistent with the guidelines and follow our coding style.

#### General guidelines and philosophy for contribution

* Include unit tests when you contribute new features, as they help to
  a) prove that your code works correctly, and b) guard against future breaking
  changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs
  usually indicates insufficient test coverage.
* When you contribute a new feature to TensorFlow, the maintenance burden is (by
  default) transferred to the SIG team. This means that benefit of the
  contribution must be compared against the cost of maintaining the feature.

#### Python coding style

Changes to Python code should conform to
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with indent width of 2 spaces.

This is enforced using [pre-commit](https://pre-commit.com/) hooks that run: `yapf`, `isort`, `pylint`.

To run the checks manually, follow [Development tips](#development-tips) and run:
```bash
pre-commit run --hook-stage push --files tfx_addons/__init__.py
```

#### License

Include a license at the top of new files.

* [Python license example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn.py#L1)

#### Testing your code

We use pytest to run tests. You can run tests locally using:

- Create virtual environemnt: `python3 -m venv env`
- Activate virtual environment: `source env/bin/activate && pip install --upgrade pip`
- Choose component to develop: `export COMPONENT_NAME=mlmd_client` (replace with the component you will be developing)
- Install test packages: `pip install -e ".[$COMPONENT_NAME,test]"`
- Run tests: `python -m pytest tfx_addons/$COMPONENT_NAME`

Note that only files that end with `_test.py` will be recognized as test. Learn more on writing pytest tests in [pytest docs](https://docs.pytest.org/en/latest/getting-started.html#create-your-first-test).
