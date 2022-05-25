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


def get_pkg_metadata():
  # Version
  context = {}
  base_dir = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(base_dir, "tfx_addons", "version.py")) as fp:
    exec(fp.read(), context)  # pylint: disable=exec-used

  return context["_PKG_METADATA"]


def get_version():
  # Version
  context = {}
  base_dir = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(base_dir, "tfx_addons", "version.py")) as fp:
    exec(fp.read(), context)  # pylint: disable=exec-used

  return context["__version__"]


def get_long_description():
  base_dir = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(base_dir, "README.md")) as fp:
    return fp.read()


TESTS_REQUIRE = ["pytest", "pylint", "pre-commit", "isort", "yapf"]

PKG_REQUIRES = get_pkg_metadata()
EXTRAS_REQUIRE = PKG_REQUIRES.copy()
EXTRAS_REQUIRE["all"] = list(
    set(itertools.chain.from_iterable(list(PKG_REQUIRES.values()))))
EXTRAS_REQUIRE["test"] = TESTS_REQUIRE

setup(
    name=PROJECT_NAME,
    version=get_version(),
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
