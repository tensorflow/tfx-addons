# SIG Addons Releases

SIG TFX Addons release process consists of the folowing steps:

The idea is to merge feature PR into a release branch, which is later merged into the main/master branch.

1. Create new rX.X branch on https://github.com/tensorflow/tfx-addons
2. Create and merge a new PR into the release branch
	* Set the correct version and suffix in [version.py](https://github.com/tensorflow/tfx-addons/blob/master/tensorflow_addons/version.py)
	* Ensure the proper minimum and maximum tested versions of TFX are set in [version.py](https://github.com/tensorflow/tfx-addons/blob/master/tfx_addons/version.py)
	* Ensure proper supported python libraries are set in [setup.py](https://github.com/tensorflow/addons/blob/master/setup.py)
3. Create and merge a new PR to merge rX.X branch to main/master
4. Publish and tag a [release on Github](https://github.com/tensorflow/tfx-addons/releases)
    * Add updates for new features, enhancements, bug fixes
    * Add contributors using `git shortlog <last-version>..HEAD -s`
    * TODO: **NOTE: This will trigger a GitHub action to release the wheels on PyPi**

## SIG Addons Release Team

Current Release Team:
TODO: TBD
