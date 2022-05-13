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
"""Demo repository for credit card transations fraud dataset
"""
from datetime import timedelta

from feast import BigQuerySource, Entity, FeatureView, ValueType

# Add an entity for users
user_entity = Entity(
    name="user_id",
    description=
    "A user that has executed a transaction or received a transaction",
    value_type=ValueType.STRING)

# Add two FeatureViews based on existing tables in BigQuery
user_account_fv = FeatureView(
    name="user_account_features",
    entities=["user_id"],
    ttl=timedelta(weeks=52),
    batch_source=BigQuerySource(
        table_ref="feast-oss.fraud_tutorial.user_account_features",
        event_timestamp_column="feature_timestamp"))

user_has_fraudulent_transactions_fv = FeatureView(
    name="user_has_fraudulent_transactions",
    entities=["user_id"],
    ttl=timedelta(weeks=52),
    batch_source=BigQuerySource(
        table_ref="feast-oss.fraud_tutorial.user_has_fraudulent_transactions",
        event_timestamp_column="feature_timestamp"))
