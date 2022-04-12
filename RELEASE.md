# SIG Addons Releases

TFX Addons follows [Semantic Versioning 2.0](https://semver.org/) strategy.

## Major/Minor releases

1. Create new `rX.Y` branch on https://github.com/tensorflow/tfx-addons from `main`.
2. Create new PR with updates to `version.py` against `rX.Y` branch.
	* Set the correct version and suffix in [version.py](https://github.com/tensorflow/tfx-addons/blob/master/tensorflow_addons/version.py).
	* Ensure the proper minimum and maximum tested versions of TFX are set in [version.py](https://github.com/tensorflow/tfx-addons/blob/master/tfx_addons/version.py).
	* Ensure proper supported python libraries are set in [setup.py](https://github.com/tensorflow/addons/blob/master/setup.py).
3. Create a [new release](https://github.com/tensorflow/tfx-addons/releases) from `rX.Y` branch. Create a tag with `vX.Y.Z` name.
    * Add updates for new features, enhancements, bug fixes
    * Add contributors using `git shortlog <last-version>..HEAD -s`

## Patch releases
1. Cherry-pick commits to `rX.Y` branch
2. Create new PR with increasing `_PATCH_VERSION` in `version.py` against `rX.Y` branch.
	* Set the correct version and suffix in [version.py](https://github.com/tensorflow/tfx-addons/blob/master/tensorflow_addons/version.py).
	* Ensure the proper minimum and maximum tested versions of TFX are set in [version.py](https://github.com/tensorflow/tfx-addons/blob/master/tfx_addons/version.py).
	* Ensure proper supported python libraries are set in [setup.py](https://github.com/tensorflow/addons/blob/master/setup.py).
3. Create a [new release](https://github.com/tensorflow/tfx-addons/releases) from `rX.Y` branch. Create a tag with `vX.Y.Z` name.
    * Add updates for new features, enhancements, bug fixes
    * Add contributors using `git shortlog <last-version>..HEAD -s`



## SIG Addons Release Team

Current Release Team:
TODO: TBD
