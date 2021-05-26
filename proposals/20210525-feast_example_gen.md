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

* At a high level, there are 3 major APIs [3]:
1. Connect to a remote Feast instance. The connection string could be part of a general procedure to configure the feature store using a JSON/YAML file.
2. Provide a wrapper for `get_online_features`. Feast returns an `OnlineResponse` [5], which will be converted to a tensor.
3. Provide a wrapper for `get_historical_features`. Feast returns a `RetrievalJob` [6], which will be used to materialize the results and convert them to a tensor.

* Important Considerations
1. We want to generalize the implementation as much as possible to make it straightforward to add support for other feature stores besides Feast in the future. To this end, it would be useful to generalize the notion of Feast's entity types [4].

2. Today Feast only supports BigQuery queries for `get_historical_features`. We should parameterize the query so that users can use other offline data stores in the future. We should propagate the error message for bad queries.

* Testing: unit and integration tests

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
2. Michal Brys, michalbrys, michal.brys@openx.com

## References
1. https://docs.feast.dev/
2. https://www.kubeflow.org/docs/external-add-ons/feature-store/overview/
3. https://rtd.feast.dev/en/latest/
4. https://github.com/feast-dev/feast/issues/405
5. https://rtd.feast.dev/en/latest/feast.html#feast.online_response.OnlineResponse
6. https://rtd.feast.dev/en/latest/#feast.feature_store.FeatureStore.get_historical_features
