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
"""The ModelCard TFX/MLMD artifact."""

import datetime

import ml_metadata as mlmd
from absl import logging
from ml_metadata import errors
from ml_metadata.proto import metadata_store_pb2
from tfx.types.artifact import Artifact
from tfx.types.system_artifacts import Metrics


class ModelCard(Artifact):
  """A [TFX/MLMD artifact](https://www.tensorflow.org/tfx/guide/mlmd#data_model) to model card assets.

  Assets include:
    * a data file containing the model card fields, located at
    `<uri>/data/model_card.proto`.
    * the model card itself, located at the `<uri>/model_card/ directory`.
  """
  TYPE_NAME = 'tfx_addons.ModelCard'
  TYPE_ANNOTATION = Metrics


def create_and_save_artifact(
    artifact_name: str, artifact_uri: str,
    store: mlmd.MetadataStore) -> metadata_store_pb2.Artifact:
  """Generates and saves a ModelCard artifact to the specified MetadataStore.

  Args:
    artifact_name: The name for the ModelCard artifact. A timestamp will be
      appended to this to distinguish model cards created from the same job.
    artifact_uri: The uri for the ModelCard artifact.
    store: The MetadataStore where the ModelCard artifact and artifact type are
      saved.

  Returns:
    The saved artifact, which can be used to store model card assets.
  """

  try:
    type_id = store.get_artifact_type(ModelCard.TYPE_NAME).id
  except errors.NotFoundError:
    type_id = store.put_artifact_type(
        metadata_store_pb2.ArtifactType(name=ModelCard.TYPE_NAME))
  name = ''.join(
      [artifact_name, '_',
       datetime.datetime.now().strftime('%H:%M:%S')])

  # Save artifact to store. Also populates the artifact's id.
  artifact_id = store.put_artifacts([
      metadata_store_pb2.Artifact(type=ModelCard.TYPE_NAME,
                                  type_id=type_id,
                                  uri=artifact_uri,
                                  name=name)
  ])[0]
  artifact = store.get_artifacts_by_id([artifact_id])[0]
  logging.info(
      'Successfully saved ModelCard artifact %s with uri=%s and id=%s.',
      artifact.name, artifact.uri, artifact.id)
  return artifact
