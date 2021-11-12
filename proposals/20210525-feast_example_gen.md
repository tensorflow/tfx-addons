#### SIG TFX-Addons
# Project Proposal

**Your name:** Badrul Chowdhury

**Your email:** badrulchowdhury17@gmail.com

**Your company/organization:** Apple

**Project name:** ExampleGen for Feast Feature Store

## Project Description

This proposal is for a new component to enable connecting to and reading from feature stores via a simple and unified API. The first iteration will only support Feast [1], but we will add support for other popular feature stores over time. Similarly, we will restrict v1 of our library to only read from feature stores- we will evaluate supporting writes later.

## Project Category
Component

## Project Use-Case(s)

Feature stores such as Feast are becoming increasingly popular as (managed) platforms to manage all feature-related information. A separate feature store has several benefits including (but not limited to) the following [2]:

1. Feature sharing and reuse
2. Serving features at scale
3. Consistency between training and serving (inference)
4. Point-in-time correctness
5. Data quality and validation

Different organizations may use different feature stores in their ML stack, each with its own APIs for reading online and historical features. This is a significant barrier to users as they will have to learn to use a different set of APIs for each feature store. Furthermore, different feature stores may return different feature vector representations. The proposed component will solve these issues:

1. It will provide a unified API across different feature stores. It will hide the mechanics of reading from different feature stores from the user.
2. It will always return the feature vector as a tensor. It will convert different representations of the feature vector to a tensor transparently from the user.

**Disclaimers**

Our organizations currently do not use the project and they have no plans to use it once it is implemented as far as we know.

There is no potential impact, overlap, or synergies with other projects. Dependencies are outlined below.

## Project Implementation

The Feast connector will be implemented as a custom `ExampleGen` component. It will inherit from `QueryBasedExampleGen`. It will ultimately closely resemble `PrestoExampleGen` [8]. `FeastExampleGen` will emit `tf.Example` records.

```python
from tfx.examples.custom_components.feast_example_gen.proto import feast_config_pb2
from tfx.examples.custom_components.feast_example_gen.feast_component.component import FeastExampleGen

feast_config = feast_config_pb2.FeastConnConfig(host=.., port=..)
example_gen = FeastExampleGen(feast_config, query='SELECT event_timestamp, order_id, customer_id from gcp_project.my_ds.customer_orders', custom_config=..)
```

The connector will switch on `custom_config` keys to choose between `get_online_features()` &#8594; `OnlineResponse` and `get_historical_features` &#8594; `RetrievalJob`.

The following are some `custom_config` keys:
* Offline store
    * `type` specifies the backend datastore.
    * `entity_df` is a parameter to `get_historical_features()` in Feast.
    * `feature_refs` is a parameter to `get_historical_features()` in Feast.

The `entity_df` key in `custom_config` will override the `query` parameter to the `FeastExampleGen` constructor.

* Online store
    * `feature_refs` is a parameter to `get_online_features()` in Feast.
    * `entity_rows` is a parameter to `get_online_features()` in Feast.

The `query` parameter to the `FeastExampleGen` constructor only applies to the Feast offline store. It will be ignored if valid online store parameters are specified in `custom_config`.

* Other Important Considerations
1. We want to generalize the implementation as much as possible to make it straightforward to add support for other feature stores besides Feast in the future. To this end, it would be useful to generalize the notion of Feast's entity types [4].

2. Today Feast only supports BigQuery queries for `get_historical_features`. We will offer `custom_config` options to specify other backend datastores. We should propagate the error message for bad queries.

3. Feast will implement `to_tf_dataset()` methods to convert to native TensorFlow data types for both `get_online_features()` &#8594; `OnlineResponse` and `get_historical_features` &#8594; `RetrievalJob`. `FeastExampleGen` will then emit `tf.Example` records based on the specified `output_config`.

* Testing: unit and integration tests

Reading from the Feast offline store fits in nicely with the `QueryBasedExampleGen` model. On the other hand, reading from the online store does NOT fit in nicely with this model and would likely involve additional complexities. **Therefore, although we include details for reading from the online store here, this proposal is committing to add support for the offline store only- the online store will be a stretch goal for this effort.** The decision to add support for the Feast offline store first is also motivated by the fact that most users today are looking to read data from Feast for training and further processing downstream.

### Languages
Python

### Releases
The code will reside in `tfx/addons/feature-store-connector`. We will also publish the  `tfx-addons-feature-store-connector` artifact to PyPi.

## Project Dependencies
| Dependency    | Version   | License       |
|-	            |-	        |-	            |
|Feast	        | 0.10.3    | Apache 2.0    |

## Project Team
1. Badrul Chowdhury, BACtaki, badrulchowdhury17@gmail.com
2. Gerard Saez, casassg, gcasassaez@twitter.com
3. Wihan Booyse, wihanbooyse, wihan@kriterion.ai
4. Neelan Pather, neelanpather, neelan@kriterion.ai

## References
1. https://docs.feast.dev/
2. https://www.kubeflow.org/docs/external-add-ons/feature-store/overview/
3. https://rtd.feast.dev/en/latest/
4. https://github.com/feast-dev/feast/issues/405
5. https://rtd.feast.dev/en/latest/feast.html#feast.online_response.OnlineResponse
6. https://rtd.feast.dev/en/latest/#feast.feature_store.FeatureStore.get_historical_features
7. https://www.tensorflow.org/tfx/api_docs/python/tfx/components/example_gen/component/QueryBasedExampleGen
8. https://github.com/tensorflow/tfx/tree/983794c2ec2ca567dddffab0e4827fa29bcb2230/tfx/examples/custom_components/presto_example_gen/presto_component