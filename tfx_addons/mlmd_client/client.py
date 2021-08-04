# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""MLMD representations for TFX Pipeline. References objects from MLMD to inspect their status.
"""

from functools import lru_cache
from typing import List, Optional

from ml_metadata.metadata_store import MetadataStore
from ml_metadata.proto import Artifact, metadata_store_pb2
from tfx.orchestration.metadata import _CONTEXT_TYPE_PIPELINE
from tfx.orchestration.pipeline import Pipeline


class _MetadataClientBase:
  def __init__(
      self, metadata_connection_config: metadata_store_pb2.ConnectionConfig):
    self._metadata_connection_config = metadata_connection_config
    self._store = None

  @property
  def _mlmd(self):
    if not self._store:
      self._store = MetadataStore(self._metadata_connection_config)
    return self._store

  @property
  def connection_config(self):
    return self._metadata_connection_config

  @property
  @lru_cache(None)
  def artifact_types(self):
    return {x.name: int(x.id) for x in self._mlmd.get_artifact_types()}


class ModelArtifact(_MetadataClientBase):
  """Context for a specific synchronous pipeline run.
    Unique pipeline_name + run_id.
    """
  def __init__(
      self,
      model_artifact: metadata_store_pb2.Artifact,
      metadata_connection_config: metadata_store_pb2.ConnectionConfig,
  ) -> None:
    super().__init__(metadata_connection_config)
    self._artifact = model_artifact

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}(pipeline_name="{self.pipeline_name}",'\
    ' run_id="{self.run_id}"")'


class PipelineContext(_MetadataClientBase):
  """Reference to pipeline context in MLMD."""
  def __init__(
      self,
      pipeline_context: metadata_store_pb2.Context,
      metadata_connection_config: metadata_store_pb2.ConnectionConfig,
  ) -> None:
    """Creates instance of PipelineContext
        Args:
            pipeline_context: MLMD Context for a pipeline.
            metadata_connection_config: MLMD connection config.
        """
    super().__init__(metadata_connection_config)
    self._context = pipeline_context

  @staticmethod
  def get_pipeline_context(mlmd_client: MetadataStore,
                           pipeline_name: str) -> metadata_store_pb2.Context:
    """Static method to get pipeline context.
        Args:
            mlmd_client: MLMD MetadataStore instance
            pipeline_name: Pipeline name to retrieve
        Returns:
            metadata_store_pb2.Context: Returns None if context doesn't
                exist in MLMD yet. This can happen if pipeline has not
                been executed yet.
        """
    return mlmd_client.get_context_by_type_and_name(_CONTEXT_TYPE_PIPELINE,
                                                    pipeline_name)

  def get_artifact_by_type_name(self, type_name: str) -> List[Artifact]:
    """Returns artifacts of a given type.

      Args:
        type_name: Name of artifact type to retrieve.
      Returns:
        List[Artifact]: List of artifacts of given type.
    """
    artifacts = self._mlmd.get_artifacts_by_context(self._context.id)
    return [
        artifact for artifact in artifacts
        if artifact.type_id == self.artifact_types.get(type_name, -1)
    ]

  @property
  def name(self) -> str:
    return self._context.name

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}(name="{self.name}")'


class MetadataClient(_MetadataClientBase):
  """MLMD read client."""
  def __init__(
      self,
      metadata_connection_config: metadata_store_pb2.ConnectionConfig = None,
  ):
    """Creates instance of MetadataClient.
        Args:
            metadata_config: Configuration to connect to MLMD.
                Defaults to staging DB connection.
        """
    super().__init__(metadata_connection_config)

  @staticmethod
  def from_pipeline(pipeline: Pipeline) -> Optional[PipelineContext]:
    """Retrieve PipelineContext given a pipeline.
        Helper method to make this recurring task easier.

        Args:
            pipeline: Pipeline to retrieve context from.
        Returns:
            Optional[PipelineContext]: Returns None if context doesn't
                exist in MLMD yet. This can happen if pipeline has not been
                executed yet.
        """
    return MetadataClient(pipeline.metadata_connection_config).get_pipeline(
        pipeline.pipeline_info.pipeline_name)

  def get_pipeline(self, pipeline_name: str) -> Optional[PipelineContext]:
    """Retrieve PipelineContext given a pipeline name.
        Args:
            pipeline_name: Name of pipeline to retrieve
        Returns:
            Optional[PipelineContext]: Returns None if context
                doesn't exist in MLMD yet. This can happen if pipeline has
                not been executed yet.
        """
    context = PipelineContext.get_pipeline_context(self._mlmd, pipeline_name)
    if not context:
      return None
    return PipelineContext(context, self.connection_config)

  def __repr__(self) -> str:
    connection_str = str(self.connection_config).replace("\n", ",")
    return f"{self.__class__.__name__}(connection_config={{{connection_str}}})"
