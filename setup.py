"""Package Setup script for TFX Addons."""
import datetime
import itertools
import os
import sys

from setuptools import find_namespace_packages, setup


def get_last_commit_time() -> str:
    string_time = os.getenv("NIGHTLY_TIME").replace('"', "")
    return datetime.strptime(string_time, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d%H%M%S")


def get_project_name_version():
    # Version
    version = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "tfx_addons", "version.py")) as fp:
        exec(fp.read(), version)

    project_name = "tfx-addons"
    return project_name, version


project_name, version = get_project_name_version()
inclusive_min_tfx_version = version["INCLUSIVE_MIN_TFX_VERSION"]
exclusive_max_tfx_version = version["EXCLUSIVE_MAX_TFX_VERSION"]


NAME = "tfx-addons"

TESTS_REQUIRE = ["pytest", "pylint", "pre-commit", "isort", "yapf"]

PKG_REQUIRES = {
    # Add dependencies here for your project. Avoid using install_requires.
    "mlmd_client": ["ml-pipelines-sdk>=1.0.0<2", "ml-metadata>=1.0.0<2"],
    "schema_curation": [
        "tfx>={},<{}".format(inclusive_min_tfx_version, exclusive_max_tfx_version),
    ],
    "xgboost_evaluator": [
        "tfx>={},<{}".format(inclusive_min_tfx_version, exclusive_max_tfx_version),
        "xgboost>=1.0.0",
    ],
    "sampler": ["tensorflow>=2.0.0"],
}
EXTRAS_REQUIRE = PKG_REQUIRES.copy()
EXTRAS_REQUIRE["all"] = list(
    set(itertools.chain.from_iterable(list(PKG_REQUIRES.values())))
)
EXTRAS_REQUIRE["test"] = TESTS_REQUIRE

setup(
    name=NAME,
    version=version["__version__"],
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
    packages=find_namespace_packages(
        include=[
            # Add here new library package
            "tfx_addons",
        ]
        + [f"tfx_addons.{m}.*" for m in PKG_REQUIRES]
    ),
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
