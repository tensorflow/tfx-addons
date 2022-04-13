# Copyright 2022 The TensorFlow Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Package Setup script for TFX Addons."""
import itertools
import os

from setuptools import find_namespace_packages, setup

PROJECT_NAME = "tfx-addons"


def get_project_version():
  # Version
  extracted_version = {}
  base_dir = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(base_dir, "tfx_addons", "version.py")) as fp:
    exec(fp.read(), extracted_version)  # pylint: disable=exec-used

  return extracted_version


def get_long_description():
  base_dir = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(base_dir, "README.md")) as fp:
    return fp.read()


version = get_project_version()
inclusive_min_tfx_version = version["INCLUSIVE_MIN_TFX_VERSION"]
exclusive_max_tfx_version = version["EXCLUSIVE_MAX_TFX_VERSION"]

TESTS_REQUIRE = ["pytest", "pylint", "pre-commit", "isort", "yapf"]

required_tfx_version = "tfx>={},<{}".format(inclusive_min_tfx_version,
                                            exclusive_max_tfx_version)
required_ml_pipelines_sdk_version = "ml_pipelines_sdk>={},<{}".format(
    inclusive_min_tfx_version, exclusive_max_tfx_version)
required_ml_metadata_version = "ml_metadata>={},<{}".format(
    inclusive_min_tfx_version, exclusive_max_tfx_version)

PKG_REQUIRES = {
    # Add dependencies here for your project. Avoid using install_requires.
    "mlmd_client":
    [required_ml_pipelines_sdk_version, required_ml_metadata_version],
    "schema_curation": [
        required_tfx_version,
    ],
    "xgboost_evaluator": [
        required_tfx_version,
        "xgboost>=1.0.0",
    ],
    "sampler": ["tensorflow>=2.0.0"],
    "message_exit_handler": [
        "kfp>=1.8,<1.9",
        "slackclient>=2.9.0",
        "pydantic>=1.8.0",
    ],
}
EXTRAS_REQUIRE = PKG_REQUIRES.copy()
EXTRAS_REQUIRE["all"] = list(
    set(itertools.chain.from_iterable(list(PKG_REQUIRES.values()))))
EXTRAS_REQUIRE["test"] = TESTS_REQUIRE

setup(
    name=PROJECT_NAME,
    version=version["__version__"],
    description="TFX Addons libraries",
    author="The Tensorflow Authors",
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url="https://github.com/tensorflow/tfx-addons",
    project_urls={
        # ToDo(gcasassaez): To add docs once we have some docs integrated.
        # "Documentation": "",
        "Bug Tracker": "https://github.com/tensorflow/tfx-addons/issues",
    },
    extras_require=EXTRAS_REQUIRE,
    tests_require=TESTS_REQUIRE,
    packages=find_namespace_packages(include=[
        # Add here new library package
        "tfx_addons",
    ] + [f"tfx_addons.{m}.*"
         for m in PKG_REQUIRES] + [f"tfx_addons.{m}" for m in PKG_REQUIRES]),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
