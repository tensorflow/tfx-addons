"""Internal script to parse changed files and potential pkgs and returns the overlap"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

logging.getLogger().setLevel(logging.INFO)


def _get_pkg_metadata():
  # Version
  context = {}
  base_dir = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  with open(os.path.join(base_dir, "tfx_addons", "version.py")) as fp:
    exec(fp.read(), context)  # pylint: disable=exec-used

  return context["_PKG_TESTABLE"]


def _get_affected_components(affected_files_manifest: str,
                             pkg_metadata: Dict[str, Any]) -> List[str]:

  with open(affected_files_manifest) as f:
    affected_files: List[str] = json.load(f)
  logging.info("Found affected files: %s", affected_files)

  components_to_run = set()
  for f in affected_files:
    if f.startswith("tfx_addons"):
      file_component = f.replace("tfx_addons/", "").split("/", maxsplit=1)[0]
      if file_component in pkg_metadata:
        logging.info("Package %s is marked for testing", file_component)
        components_to_run.add(file_component)
      else:
        logging.warning(
            "Package %s is not in _PKG_TESTABLE variable for version.py",
            file_component)
  return list(components_to_run)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_manifest")
  args = parser.parse_args()
  affected_components = _get_affected_components(args.file_manifest,
                                                 _get_pkg_metadata())
  print(f"::set-output name=components::{json.dumps(affected_components)}")
