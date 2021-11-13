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
"""Losely based on internal examples and Feast tutorial: https://docs.feast.dev/v/v0.6-branch/user-guide/feature-retrieval

This is just a prototype meant to be used as reference.
"""

import tfx.v1 as tfx
from feast.infra.offline_stores.bigquery import BigQueryOfflineStoreConfig
from feast.repo_config import RepoConfig
from pydantic import BaseModel

from tfx_addons.feast_examplegen import FeastExampleGen


# Use pydantic here to define this large configuration class (mostly out of easiness to use vs using protobuf)
class FeastConnConfig(BaseModel):
  repo_config: RepoConfig
  gcp_project: str


# This is a pydantic model, so it can be serialized into JSON in the component and reloaded back into the Python object in the executor!
conn_config = FeastConnConfig(
    gcp_project="some_project",
    repo_config=RepoConfig(
        registry="gs://feast_origin_registry/registry.db",
        project="production",
        provider="local",
        offline_store=BigQueryOfflineStoreConfig(type="bigquery")))

# Component definition
example_gen = FeastExampleGen(
    conn_config=conn_config,
    entity_query=
    "SELECT * FROM `production.feast_origin.feast_origin_v1.user` WHERE datetime>=@begin_timestamp AND datetime<=@end_timestamp",  # we use the same driver as BQExampleGen here to substitute the timestamp
    range_config=tfx.proto.StaticRange(start_span_number=0, end_span_number=1),
    # not sure if this is needed, but I believe it may?
    feature_refs=[
        'partner',
        'daily_transactions',
        'customer_feature_set:dependents',
        'customer_feature_set:has_phone_service',
    ] + ['churn']  # target
)

## Outputs
# example_gen.outputs['examples']  # TFExample output dataset (for now only tf.example, we can support other formats later)
# example_gen.outputs['statistics']  # Feast also can compute TFDV statitics. Question here is: Does feast try to load dataframe locally to perform statistics? If so we shouldnt use this. Otherwise we should use this!
# example_gen.outputs['schema']  #Feast can generate some form of schema. Can we generate this? otherwise we can do this in a separate component.