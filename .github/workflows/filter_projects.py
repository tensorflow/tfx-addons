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
"""Internal script to parse changed files and potential pkgs and returns the overlap"""

import argparse
import json
import logging
import os
from typing import List

logging.getLogger().setLevel(logging.INFO)

# NB(casassg): Files that if changed should trigger running CI for all projects.
# This are files which are core and we want to avoid causing outages
# because of them
RUN_ALL_FILES = [
    "tfx_addons/version.py", "setup.py", ".github/workflows/ci.yml",
    "pyproject.toml"
]

# Get event that triggered workflow
# See: https://docs.github.com/en/actions/learn-github-actions/environment-variables#default-environment-variables
GH_EVENT_NAME = os.environ.get("GITHUB_EVENT_NAME", "unknown")


def _get_testable_projects() -> List[str]:
  """Get _PKG_METADATA from version.py which contains what projects are active
  """
  context = {}
  base_dir = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  with open(os.path.join(base_dir, "tfx_addons", "version.py")) as fp:
    exec(fp.read(), context)  # pylint: disable=exec-used

  return list(context["_PKG_METADATA"].keys())


def get_affected_projects(affected_files: List[str]) -> List[str]:
  """Given a list of affected files, and  projects that can be tested,
  find what projects should CI run"""

  logging.info("Found affected files: %s", affected_files)
  testable_projects = _get_testable_projects()
  if GH_EVENT_NAME == "push":
    logging.info("GitHub Action trigger is %s, running all projects",
                 GH_EVENT_NAME)
    return testable_projects
  else:
    logging.info("GitHub Action trigger is %s, filtering projects",
                 GH_EVENT_NAME)
  for run_all_file in RUN_ALL_FILES:
    if run_all_file in affected_files:
      logging.warning("Found change in %s, running all projects", run_all_file)
      return testable_projects
  projects_to_test = set()
  for file in affected_files:
    if file.startswith("tfx_addons"):
      file_component = file.replace("tfx_addons/", "").split("/",
                                                             maxsplit=1)[0]
      if file_component in testable_projects:
        logging.info("Package %s is marked for testing", file_component)
        projects_to_test.add(file_component)
      else:
        logging.warning(
            "Package %s is not in _PKG_TESTABLE variable for version.py",
            file_component)
  return list(projects_to_test)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_manifest")

  args = parser.parse_args()

  with open(args.file_manifest, "r") as f:
    affected_components = get_affected_projects(json.load(f))
  print(json.dumps(affected_components))
