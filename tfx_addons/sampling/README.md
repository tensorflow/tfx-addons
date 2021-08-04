#### SIG TFX-Addons

# Project Proposal

------

**Your name:** Daniel Kim

**Your email:** danielk@twitter.com

**Your company/organization:** Twitter

**Project name:** Sampling Component

## Project Description

This project will be a fully custom component that inputs an artifact in `tfRecord` format of `tf.Example`s and randomly undersamples or randomly oversamples it, reducing the data to the lowest- or highest-frequency class. It will primarily use an underlying Apache Beam pipeline that will be wrapped inside the TensorFlow component.

## Project Category

Component

## Project Use-Case(s)

As this project represents a very general operation used widely in machine learning data processing, we anticipate that it will have wide-ranging use cases, the most evident being in cases where dependent variable classes have wildly different relative frequencies and under/oversampling is needed to help effectively train a classifier. The potential impact will likely be large due to this, and our organization will likely utilize this project in the future.

## Project Solutions

We considered multiple possible solutions and implementations for this project before deciding on an Apache Beam-based pipeline, including standard Python code and the utilization of a BigQuery query in order to perform the random /oversampling task. Using a pure Python-based algorithm with `multiprocessing` will likely be inefficient for the purposes of a parallelizable computation such as this one, and utilizing solutions such as Dask would introduce unnecessary dependencies into our project. 

BigQuery is also a very good option, and a great fallback in case Apache Beam turns out to be infeasible for this project, but Apache Beam has better python integration through custom `DoFn`s that may help us with our implementation of other algorithms later on. In this case, we would load the data in and out of a BigQuery table and perform our operations within this table. The component would then either utilize a schema generated from `SchemaGen` or inflect one on its own, potententially adding an unneeded dependency into the component or performing unnecessary inflection.

## Project Implementation

At a high level, the plan is to use Apache Beam to ingest a `tfRecord` of `tf.Examples`, shuffle them, convert them into a key-value `PCollection` with keys as class values and values as data points, and then perform the actual under/oversampling. Null values (and values that have key classes that are specified by the user) will not be part of the over/undersampling step; they will be separated and added back into the sampled dataset. The algorithm will be written as an Apache Beam pipeline, which will be wrapped into a TensorFlow custom component (with custom executor and spec) to use with TFX pipelines. The component would be written as inputting a `TFRecord` artifact of `tf.Examples` and exporting a similar `TFRecord` artifact, making its placement in a pipeline nearly ubiquitous. 

Later additions to the project could include the integration through Apache Beam of one or more other, more complex undersampling or ovesampling algorithms. Our likely focus would be SMOTE for oversampling and either ENN or Tomek Links for undersampling. These would likely be implemented as custom Python functions within the Apache Beam pipeline, although the focus for now is currently the initial random sampling component.

## Project Dependencies

tensorflow, TFX, Apache Beam

## Project Team

List the members of the project team. Include their names, Github user IDs, and email addresses. Identify project leaders.

* Daniel Kim, kindalime, danielk@twitter.com