# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Internal script to parse changed files and potential examples and returns the overlap"""

import argparse
import json
import logging
import os
import sys
from typing import List

import pkg_resources

# Dynamically load .github as module so that we can do relative import here
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, ".github"))

from workflows import filter_projects  # pylint: disable=wrong-import-position

logging.getLogger().setLevel(logging.INFO)

# Get event that triggered workflow
# See: https://docs.github.com/en/actions/learn-github-actions/environment-variables#default-environment-variables
GH_EVENT_NAME = os.environ.get("GITHUB_EVENT_NAME", "unknown")

# NB(casassg): Files that if changed should trigger running CI for all examples.
# This are files which are core and we want to avoid causing outages
# because of them
RUN_ALL_FILES = [
    ".github/workflows/ci_examples.yml",
]


def _get_testable_examples() -> List[str]:
  """Get projects that have requirements.txt.
  """

  projects = []
  for project in os.listdir(os.path.join(BASE_DIR, "examples")):
    if not os.path.exists(
        os.path.join(BASE_DIR, "examples", project, "requirements.txt")):
      continue
    test_files = [
        f for f in os.listdir(os.path.join(BASE_DIR, "examples", project))
        if 'test' in f and f.endswith(".py")
    ]
    if test_files:
      projects.append(project)

  return projects


def _get_affected_examples(affected_files: List[str]) -> List[str]:
  """Given a list of affected files, and  projects that can be tested,
  find what projects should CI run"""

  logging.info("Found affected files: %s", affected_files)
  testable_examples = _get_testable_examples()
  logging.info("Found %s testable example folders", testable_examples)
  if GH_EVENT_NAME == "push":
    logging.info("GitHub Action trigger is %s, running all projects",
                 GH_EVENT_NAME)
    return testable_examples
  else:
    logging.info("GitHub Action trigger is %s, filtering projects",
                 GH_EVENT_NAME)
  for run_all_file in RUN_ALL_FILES:
    if run_all_file in affected_files:
      logging.warning("Found change in %s, running all projects", run_all_file)
      return testable_examples

  examples_to_test = set()
  for file in affected_files:
    if file.startswith("examples"):
      file_component = file.replace("examples/", "").split("/", maxsplit=1)[0]
      if file_component in testable_examples:
        logging.info("Package %s is marked for testing", file_component)
        examples_to_test.add(file_component)
      else:
        logging.warning("Example %s is not testable, skipping", file_component)
  affected_tfxa_projects = filter_projects.get_affected_projects(
      affected_files)
  for project in testable_examples:
    with open(os.path.join(BASE_DIR, "examples", project,
                           "requirements.txt")) as f2:
      requirements = [
          l.replace("\n", "").replace("../..", "tfx_addons")
          for l in f2.readlines()
      ]
    logging.info("Found %s requirements for example %s", requirements, project)
    for req in pkg_resources.parse_requirements(requirements):
      if req.unsafe_name == "tfx_addons":
        for extra in req.extras:
          if extra in affected_tfxa_projects:
            logging.info("Example %s depends on tfx_addons.%s running in CI.",
                         project, extra)
            examples_to_test.add(project)
  return list(examples_to_test)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_manifest")

  args = parser.parse_args()

  with open(args.file_manifest, "r") as f:
    affected_components = _get_affected_examples(json.load(f))
  print(json.dumps(affected_components))
