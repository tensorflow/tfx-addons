#### SIG TFX-Addons
# Project Proposal for Upload Predictions to BigQuery

**Your name:** Hannes Max Hapke

**Your email:** hannes@digits.com

**Your company/organization:** Digits Financial, Inc.

**Project name:** Upload Predictions to BigQuery component

## Project Description

The project addresses the project idea #78. The TFX `BulkInferrer` allows pipeline to apply an ML model (loaded or trained in the pipeline) and generates predictions for the provided inference data.

This project will provide a component which receives the predictions from the `BulkInferrer` and writes the results to BigQuery.

## Project Category

Component

## Project Use-Case(s)

Such a component is useful for generating predictions within the pipeline or for two-step pipelines producing semi-supervised ML models.

## Project Implementation

The existing implementation was written as a "traditional" TFX component with its `ComponentSpec`, `Executor`, etc. to run efficiently on Apache Beam.

The implementation receives 3 artifacts:
* transform_graph
* inference_results
* schema

The `transform_graph` is used to convert classification probabilities to a label. The TFX `schema` is used to generate the BigQuery schema for the table inserts. And the `inference_results` contain the information provided from the upstream `BulkInferrer` component.

In addition, the component accepts a number of parameters to customize the BigQuery inserts:
* bq_table_name - Table name
* filter_threshold - threshold to filter results with low confidence
* table_suffix - suffix for daily inferences
* table_partitioning - BQ partitioning setting for newly created tables
* expiration_time_delta - BQ expiration time after which the table will expire

The component processes the inference results, converts the class likelihoods into class labels, and then generates a tables schema from the TFX schema information, before it writes the information to Big Query.

The writing to Big Query is done via Apache Beam.

```
with self._make_beam_pipeline() as pipeline:
    _ = (pipeline
            | 'Read Prediction Log' >> beam.io.ReadFromTFRecord(
                prediction_log_path,
            coder=prediction_log_decoder)
            | 'Filter and Convert to Dict' >> beam.ParDo(
                FilterPredictionToDictFn(
                    labels=labels,
                    features=features,
                    ts=ts,
                    filter_threshold=exec_properties['filter_threshold'],
                )
            )
            | 'Write Dict to BQ' >> beam.io.gcp.bigquery.WriteToBigQuery(
                table=bq_table_name,
                schema=bq_schema,
                additional_bq_parameters=_ADDITIONAL_BQ_PARAMETERS,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE)
            )
```

After the completion of the datat insert, the component returns the `generated_bq_table_name` as a string artifact for downstream components.

## Project Dependencies

The component requires:
* TFX version >= 1.0.0
* Apache Beam
* TensorFlow Transform

The component implicitly requires Google Cloud as a Dependency due to the writing operation to BigQuery.

## Project Team

* Hannes Hapke (@hanneshapke), hannes -at- digits.com
* Ukjae Jeong (@jeongukjae)

# Note

Please be aware of the processes and requirements which are outlined here:

* [SIG-TFX-Addons](https://github.com/tensorflow/tfx-addons)
* [Contributing Guidelines](https://github.com/tensorflow/tfx-addons/blob/main/CONTRIBUTING.md)
* [TensorFlow Code of Conduct](https://github.com/tensorflow/tfx-addons/blob/main/CODE_OF_CONDUCT.md)
