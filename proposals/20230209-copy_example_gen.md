#### SIG TFX-Addons
# Project Proposal for CopyExampleGen component

**Your name:** Alexander Ho

**Your email:** alexanderho@google.com

**Your company/organization:** Google

**Project name:** CopyExampleGen component

## Project Description
CopyExampleGen will allow the user to copy a pre-existing Tfrecord dataset or raw data and ingest it into the pipeline, ultimately skipping the process of shuffling and running the Beam job. This process will require a dict input with split_names and their respective URI. This will output an Examples Artifact (same as the Artifact output from the ExampleGen component)  in which downstream components can use.

## Project Category
Component

## Project Use-Case(s)
CopyExampleGen will be able to do two things. First, this component will allow the user to add a dict input with split_names as the key and their respective raw data URIs as their value, then convert the data into Tfrecords, then format the directory structure so that it matches that of an Example Artifact. Second: this component will allow the user to add a dict input with split_names as the key and their respective pre-existing Tfrecords URIs as their value, then format the director structure so that it matches that of an Example Artifact.

Currently, ingesting data with the ExampleGen requires a Beam job to be ran and for the data to be shuffled. This component will save users hours/ days of having to create a workaround fully custom ExampleGen component. Some challenges our users have had:
Reshuffle doesn't work well with DirectRunner and causes OOMing. Users have been patching out shuffling in every release and doing it in the DB query. They have given up on Beam based ExampleGen and have created an entire custom ExampleGen that reads from the database and doesn’t use Beam. Link.
When the use case is a time series problem using sliding windows, shuffling before splitting in train and eval set is counterproductive as the user would need a coherent training set. Link.
Almost impossible to use ExampleGen based components for large datasets. Without it, Beam knows how to write to disk after transforming from input format to output format, allowing it to transform (slowly) large datasets that would otherwise not fit into memory. Link.

## Project Implementation
Use case #1 - Tfrecords as input URIs:
This component will:
1. Accept a dict i.e. {'split_name1': './path/to/split_name1/tfrecord1', 'split_name2': './path/to/split_name2/tfrecord2'}
2. Create an Examples Artifact with the correct directory structure
3. Register the Examples Artifact into MLMD

Use case #2 - Raw data as input URIs
This component will:
1. Accept a dict i.e. {'split_name1': './path/to/split_name1/raw_data', 'split_name2': './path/to/split_name2/raw_data'}
2. Convert these files to tfrecords
  - Register under MLMD
3. Format tfrecords to Examples Artifact directory structure
  - Register under MLMD

## Project Dependencies
Using: Python 3.8.2, Tensorflow 2.11.0, TFX 1.12.0

## Project Team
Alex Ho, alexanderho@google.com, @alxndrnh

