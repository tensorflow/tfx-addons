#### SIG TFX-Addons
# Project Proposal

**Your name:** Woosung Song

**Your email:** wssong@google.com

**Your company/organization:** Google

**Project name:** Apache Airflow for Pipeline Orchestration

## Project Description
Apache Airflow for pipeline orchestration is going to be migrated from the
official TFX to Addons.

## Project Category
Other (Orchestration)

## Project Use-Case(s)
In order to simplify core TFX for users who are not using Airflow, we would like
to separate out support for the Airflow orchestrator into a pluggable module and
make it available through TFX-Addons. This will help simplify the core TFX
install, dependencies, and tests, and decrease the size of the installed
payload.

The functionality of the orchestrator will be retained, but users will need to
update the import paths. To make the transition smoother, it will coexist on
both the official TFX and Addons for a while, and the official one will be
deprecated from the 1.14.0 release.

## Project Implementation
The basic implementation and API signatures will follow the original methods,
but the internal dependencies and testing will be reimplemented.

The import path will be moved from `tfx.orchestration.airflow` to
`tfx_addons.airflow_orchestration`.

```python
from tfx_addons.airflow_orchestration import airflow_dag_runner

def _create_pipeline():
  ...
  return [example_gen, statistics_gen, trainer, evaluator, pusher]

runner = airflow_dag_runner.AirflowDagRunner(_airflow_dag_config)
result = runner.run(_create_pipeline())
```

## Project Dependencies
It introduces `apache-airflow[mysql]>=1.10.14,<3` as the dependencies.

## Project Team
**Project Leader** : Woosung Song, lego0901, wssong@google.com
1. Woosung Song, wssong@google.com, @wssong
