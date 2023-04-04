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
"""HuggingFace Pusher runner module.
This module handles the workflow to publish
machine learning model to HuggingFace Hub.
"""
import mimetypes
import tempfile
from typing import Any, Dict, Optional, Text

import tensorflow as tf
from absl import logging
from huggingface_hub import HfApi, Repository
from requests.exceptions import HTTPError
from tfx.utils import io_utils

_MODEL_REPO_KEY = "MODEL_REPO_ID"
_MODEL_URL_KEY = "MODEL_REPO_URL"
_MODEL_VERSION_KEY = "MODEL_VERSION"

_DEFAULT_MODEL_REPO_PLACEHOLDER_KEY = "$MODEL_REPO_ID"
_DEFAULT_MODEL_URL_PLACEHOLDER_KEY = "$MODEL_REPO_URL"
_DEFAULT_MODEL_VERSION_PLACEHOLDER_KEY = "$MODEL_VERSION"


def _is_text_file(path):
  """check if a file in the given path is text type"""
  mimetype = mimetypes.guess_type(path)
  if mimetype[0] is not None:
    return 'text' in mimetype[0]
  return False


def _replace_placeholders_in_files(root_dir: str,
                                   placeholder_to_replace: Dict[str, str]):
  """Recursively open every files under the root_dir, and then
    replace special tokens with the given values in placeholder_
    to_replace. Only the contents inside text based files should
    be replaced, so the file type is checked by _is_text_file."""
  files = tf.io.gfile.listdir(root_dir)
  for file in files:
    path = tf.io.gfile.join(root_dir, file)

    if tf.io.gfile.isdir(path):
      _replace_placeholders_in_files(path, placeholder_to_replace)
    else:
      _replace_placeholders_in_file(path, placeholder_to_replace)


def _replace_placeholders_in_file(filepath: str,
                                  placeholder_to_replace: Dict[str, str]):
  """replace special tokens with the given values in placeholder_
    to_replace. This function gets called by _replace_placeholders
    _in_files function."""
  if _is_text_file(filepath):
    with tf.io.gfile.GFile(filepath, "r") as f:
      source_code = f.read()

    for placeholder in placeholder_to_replace:
      source_code = source_code.replace(placeholder,
                                        placeholder_to_replace[placeholder])

    with tf.io.gfile.GFile(filepath, "w") as f:
      f.write(source_code)


def _replace_placeholders(
    target_dir: str,
    placeholders: Dict[str, str],
    model_repo_id: str,
    model_repo_url: str,
    model_version: str,
):
  """set placeholder_to_replace before calling _replace_placeholde
    rs_in_files function"""

  if placeholders is None:
    placeholders = {
        _MODEL_REPO_KEY: _DEFAULT_MODEL_REPO_PLACEHOLDER_KEY,
        _MODEL_URL_KEY: _DEFAULT_MODEL_URL_PLACEHOLDER_KEY,
        _MODEL_VERSION_KEY: _DEFAULT_MODEL_VERSION_PLACEHOLDER_KEY,
    }

  placeholder_to_replace = {
      placeholders[_MODEL_REPO_KEY]: model_repo_id,
      placeholders[_MODEL_URL_KEY]: model_repo_url,
      placeholders[_MODEL_VERSION_KEY]: model_version,
  }
  _replace_placeholders_in_files(target_dir, placeholder_to_replace)


def _replace_files(src_path, dst_path):
  """replace the contents(files/folders) of the repository with the
    latest contents"""

  not_to_delete = [".gitattributes", ".git"]

  inside_root_dst_path = tf.io.gfile.listdir(dst_path)

  for content_name in inside_root_dst_path:
    content = f"{dst_path}/{content_name}"

    if content_name not in not_to_delete:
      if tf.io.gfile.isdir(content):
        tf.io.gfile.rmtree(content)
      else:
        tf.io.gfile.remove(content)

  inside_root_src_path = tf.io.gfile.listdir(src_path)

  for content_name in inside_root_src_path:
    content = f"{src_path}/{content_name}"
    dst_content = f"{dst_path}/{content_name}"

    if tf.io.gfile.isdir(content):
      io_utils.copy_dir(content, dst_content)
    else:
      tf.io.gfile.copy(content, dst_content)


def _create_remote_repo(access_token: str,
                        repo_id: str,
                        repo_type: str = "model",
                        space_sdk: str = None):
  """create a remote repository on HuggingFace Hub platform. HTTPError
    exception is raised when the repository already exists"""

  logging.info(f"repo_id: {repo_id}")
  try:
    HfApi().create_repo(
        token=access_token,
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk=space_sdk,
    )
  except HTTPError:
    logging.warning(
        f"this warning is expected if {repo_id} repository already exists")


def _clone_and_checkout(repo_url: str,
                        local_path: str,
                        access_token: str,
                        version: Optional[str] = None) -> Repository:
  """clone the remote repository to the given local_path"""

  repository = Repository(local_dir=local_path,
                          clone_from=repo_url,
                          use_auth_token=access_token)

  if version is not None:
    repository.git_checkout(revision=version, create_branch_ok=True)

  return repository


def _push_to_remote_repo(repo: Repository,
                         commit_msg: str,
                         branch: str = "main"):
  """push any changes to the remote repository"""

  repo.git_add(pattern=".", auto_lfs_track=True)
  repo.git_commit(commit_message=commit_msg)
  repo.git_push(upstream=f"origin {branch}")


def deploy_model_for_hf_hub(
    username: str,
    access_token: str,
    repo_name: str,
    model_path: str,
    model_version: str,
    space_config: Optional[Dict[Text, Any]] = None,
) -> Dict[str, str]:
  """Executes ML model deployment workflow to HuggingFace Hub. Refer to the
    HFPusher component in component.py for generic description of each parame
    ter. This docstring only explains how the workflow works.

    step 1. push model to the Model Hub
    step 1-1.
        create a repository on the HuggingFace Hub. if there is an existing r
        epository with the given repo_name, that rpository will be overwritten.
    step 1-2.
        clone the created or existing remote repository to the local path. Al
        so, create a branch named with model version.
    step 1-3.
        remove every files under the cloned repository(local), and copies the
        model related files to the cloned local repository path.
    step 1-4.
        push the updated repository to the given branch of remote Model Hub.

    step 2. push application to the Space Hub
    step 2-1.
        create a repository on the HuggingFace Hub. if there is an existing r
        epository with the given repo_name, that rpository will be overwritten.
    step 2-2.
        copies directory where the application related files are stored to a
        temporary directory. Since the files could be hosted in GCS bucket, t
        his process ensures every necessary files are located in the local fil
        e system.
    step 2-3.
        replacek speical tokens in every files under the given directory.
    step 2-4.
        clone the created or existing remote repository to the local path.
    step 2-5.
        remove every files under the cloned repository(local), and copies the
        application related files to the cloned local repository path.
    step 2-6.
        push the updated repository to the remote Space Hub. note that the br
        anch is always set to "main", so that HuggingFace Space could build t
        he application automatically when pushed.
    """
  outputs = {}

  # step 1
  repo_url_prefix = "https://huggingface.co"
  repo_id = f"{username}/{repo_name}"
  repo_url = f"{repo_url_prefix}/{repo_id}"

  # step 1-1
  _create_remote_repo(access_token=access_token, repo_id=repo_id)
  logging.info(f"remote repository at {repo_url} is prepared")

  # step 1-2
  local_path = "hf_model"
  repository = _clone_and_checkout(
      repo_url=repo_url,
      local_path=local_path,
      access_token=access_token,
      version=model_version,
  )
  logging.info(
      f"remote repository is cloned, and new branch {model_version} is created"
  )

  # step 1-3
  _replace_files(model_path, local_path)
  logging.info(
      "current version of the model is copied to the cloned local repository")

  # step 1-4
  _push_to_remote_repo(
      repo=repository,
      commit_msg=f"updload new version({model_version})",
      branch=model_version,
  )
  logging.info("updates are pushed to the remote repository")

  outputs["repo_id"] = repo_id
  outputs["branch"] = model_version
  outputs["commit_id"] = f"{repository.git_head_hash()}"
  outputs["repo_url"] = repo_url

  # step 2
  if space_config is not None:
    if "app_path" not in space_config:
      raise RuntimeError("the app_path is not provided. "
                         "app_path is required when space_config is set.")

    model_repo_id = repo_id
    model_repo_url = repo_url

    if "repo_name" in space_config:
      repo_id = f"{username}/{repo_name}"
      repo_url = f"{repo_url_prefix}/{repo_id}"
    else:
      repo_url = f"{repo_url_prefix}/spaces/{repo_id}"
    app_path = space_config["app_path"]
    app_path = app_path.replace(".", "/")

    # step 2-1
    _create_remote_repo(
        access_token=access_token,
        repo_id=repo_id,
        repo_type="space",
        space_sdk=space_config["space_sdk"]
        if "space_sdk" in space_config else "gradio",
    )

    # step 2-2
    tmp_dir = tempfile.gettempdir()
    io_utils.copy_dir(app_path, tmp_dir)

    # step 2-3
    _replace_placeholders(
        target_dir=tmp_dir,
        placeholders=space_config["placeholders"]
        if "placeholders" in space_config else None,
        model_repo_id=model_repo_id,
        model_repo_url=model_repo_url,
        model_version=model_version,
    )

    # step 2-4
    local_path = "hf_space"
    repository = _clone_and_checkout(
        repo_url=repo_url,
        local_path=local_path,
        access_token=access_token,
    )

    # step 2-5
    _replace_files(tmp_dir, local_path)

    # step 2-6
    _push_to_remote_repo(
        repo=repository,
        commit_msg=f"upload {model_version} model",
    )

    outputs["space_url"] = repo_url

  return outputs
