from setuptools import setup
import tfx_addons as tfxa
import itertools

NAME = "tfx-addons"
# VERSION = .... Change the version in tfx_addons/__init__.py

TESTS_REQUIRE = ["pytest", "pylint", "black", "pre-commit"]

EXTRAS_REQUIRE = {
    # Add dependencies here for your project. Avoid using install_requires.
    "mlmd_client": ["ml-pipelines-sdk>=0.26.3<1", "ml-metadata>=0.26.3<1"]
}
EXTRAS_REQUIRE["all"] = list(
    set(
        itertools.chain.from_iterable(
            list(EXTRAS_REQUIRE.values()) + [TESTS_REQUIRE])))

setup(
    name=NAME,
    version=tfxa.__version__,
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
    packages=[
        # Add here new library package
        "tfx_addons",
        "tfx_addons.mlmd_client",
    ],
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
