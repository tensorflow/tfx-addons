"""Package Setup script for TFX Addons."""
import itertools
import os
import re

from setuptools import find_namespace_packages, setup


def _get_version():
  version_file = os.path.join(os.path.dirname(__file__),
                              'tfx_addons/__init__.py')
  with open(version_file, 'r') as fp:
    version_file_text = fp.read()

  version_match = re.search(
      r"^__version__ = ['\"]([^'\"]*)['\"]",
      version_file_text,
      re.M,
  )
  if version_match:
    return version_match.group(1)
  else:
    raise RuntimeError("Unable to find version string.")


NAME = "tfx-addons"
# VERSION = .... Change the version in tfx_addons/__init__.py

TESTS_REQUIRE = ["pytest", "pylint", "pre-commit", "isort", "yapf"]

EXTRAS_REQUIRE = {
    # Add dependencies here for your project. Avoid using install_requires.
    "mlmd_client":
    ["ml-pipelines-sdk>=1.0.0,<2.0.0", "ml-metadata>=1.0.0,<2.0.0"],
    "schema_curation": [
        "tfx>=0.26.3,<2.0.0",
    ],
    "xgboost_evaluator": [
        "tfx>=1.0.0,<2.0.0",
        "xgboost>=1.0.0",
    ],
    "sampler": ["tensorflow>=2.0.0"],
    "feast_examplegen": [
        "tfx>=1.4.0,<2.0.0",
        "feast>=0.16.0,<1.0.0",
    ],
}
EXTRAS_REQUIRE["all"] = list(
    set(itertools.chain.from_iterable(list(EXTRAS_REQUIRE.values()))))
EXTRAS_REQUIRE["test"] = TESTS_REQUIRE

setup(
    name=NAME,
    version=_get_version(),
    description="TFX Addons libraries",
    author="The Tensorflow Authors",
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
        "tfx_addons"
    ] + [f'tfx_addons.{k}' for k in EXTRAS_REQUIRE]),
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
