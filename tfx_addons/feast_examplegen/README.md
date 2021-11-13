# tfx-addons-feast-examplegen
Examplegen for Feast Feature Store

## (WIP) Setup

### MacOS (old)

```
python -m venv venv
. venv/bin/activate
pip install -e .
```

### MacOS M1
Ensure you have Python 3.8 installed: 3.7 won't work on M1 and TFX won't work with 3.9.

```
brew install python@3.8
brew link --force python@3.8

which pip
```

### Install TFX on MacOS M1
Adpated from: https://towardsdatascience.com/installing-tensorflow-on-the-m1-mac-410bb36b776

```
pip install six absl-py numpy google google-api-python-client wrapt opt-einsum gast astunparse termcolor flatbuffers

pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl

--

python -m pip install tensorflow-macos

```
