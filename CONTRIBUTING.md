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

### Contributing code

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
### Development tips

We use [pre-commit](https://pre-commit.com/) to validate our code before we push to the repository. We use push over commit to allow more flexibility.

Here's how to install it locally:
- Create virtual environemnt: `python3 -m venv env`
- Activate virtual environment: `source env/bin/activate`
- Upgrade pip: `pip install --upgrade pip`
- Install test packages: `pip install -e ".[test]"`
- Install pre-commit hooks for push hooks: `pre-commit install pre-push`
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

Pre-commit will add the files for you if installed.

```bash
pre-commit run --hook-stage push --files tfx_addons/__init__.py
```

#### Testing your code

We use pytest to run tests. You can run tests locally using:

- Create virtual environemnt: `python3 -m venv env`
- Activate virtual environment: `source env/bin/activate`
- Install test packages: `pip install -e ".[all,test]"`
- Run tests: `pytest`
