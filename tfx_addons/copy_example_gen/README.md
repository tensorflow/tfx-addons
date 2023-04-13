#### SIG TFX-Addons
# Project Proposal for CopyExampleGen component

**Your name:** Alexander Ho

**Your email:** alexanderho@google.com

**Your company/organization:** Google

**Project name:** CopyExampleGen component

## Project Description
CopyExampleGen will allow the user to copy pre-existing tfrecords and ingest it into the pipeline as examples, ultimately skipping the process of shuffling and running the Beam job that is in the standard component, ExampleGen. This process will require a dict input with split names as keys and their respective URIs as the value from the user. Following suit, the component will set the artifact’s properties, generate output dict, and register contexts and execution for downstream components to use. Lastly, tfrecord file(s) in uri must resemble same `.gz` file format as the output of ExampleGen component.

Example of pipeline component definition:
```python
tfrecord_dict : Dict[str, str] = {
  "train" : "gs://path/to/tfrecords/examples/Split-train/",
  "eval" : "gs://path/to/tfrecords/examples/Split-eval/"
}

 copy_example_gen = component.CopyExampleGen(
      input_dict = json.dumps(tfrecords_dict)
 )
```

As of April 10th, 2023, tfx.dsl.components.Parameter only supports primitive types therefore, in order to properly use CopyExampleGen, the 'input_dict' of type Dict[str, str] needs to be converted into a JSON str. We can do this by simply using `json.dumps()` by adding 'tfrecords_dict' in as an argument.


## Project Category
Addon Component

## Project Use-Case(s)
CopyExampleGen will replace ExampleGen when tfrecords and split names are already in the possession of the user. Hence, a Beam job will not be run nor will the tfrecords be shuffled and/ or randomized saving data ingestion pipeline process time.

Currently, ingesting data with the ExampleGen component does not provide a way to split without random data shuffling and always runs a beam job. This component will save significant time (hours for large amounts of data) per pipeline run when a pipeline run does not require data to be shuffled. Some challenges users have had:

  1. “Reshuffle doesn't work well with DirectRunner and causes OOMing. Users have been patching out shuffling in every release and doing it in the DB query. They have given up on Beam based ExampleGen and have created an entire custom ExampleGen that reads from the database and doesn’t use Beam”.

  2. “When the use case is a time series problem using sliding windows, shuffling before splitting in train and eval set is counterproductive as the user would need a coherent training set”.


## Project Implementation
### Component

Custom Python function component: CopyExampleGen

 - `input_json_str`: will be the input parameter for CopyExampleGen of type `tfx.dsl.components.Parameter[str]`, where the user will assign their Dict[str, str] input, tfrecords_dict. However, because Python custom component development only supports primitive types, we must assign `input_json_str` to `json.dumps(tfrecords_dict)` and place the tfrecords_dict in as an argument.

 - `output_example`: Output artifact can be referenced as an object of its' specified type ArtifactType in the component function being declared. For example, if the ArtifactType is Examples, one can reference properties in an Examples ArtifactType (span, version, split_names, etc.) by calling the OutputArtifact object. This will be the variable we reference to build and register our Examples Artifact after pasrsing the tfrecords_dict input.


### Python Custom Component Implementation Details

  Using fileio.mkdir and fileio.copy, the component will then create a directory folder for each name in `split_name`. Following the creation of the `Split-name` folder, the files in the uri path will then be copied into the designated `Split-name` folder.

  Thoughts from original implementation in phase 1:
  This step can possibly use the [importer.generate_output_dict](https://github.com/tensorflow/tfx/blob/f8ce19339568ae58519d4eecfdd73078f80f84a2/tfx/dsl/components/common/importer.py#L153) function:
  Create standard ‘output_dict’ variable. The value will be created by calling the worker function. If file copying is done before this step, this method can probably be used as is to register the artifact.

  Using the keys and values from `tfrecords_dict`:
  Parse the input_dict.keys() to a str to resemble the necessary format of property `split-names` i.e. '["train","eval"]'
 
## Possible Future Development Directions
  1. There's a few open questions about how the file copying should actually done. Where does the copying that importer does actually happen? And what's the best way to change that? Are there other ways in TFX to do copying in a robust way? Maybe something in tfx.io? If there's an existing method, what has to happen in the `parse_tfrecords_dict`. Depending on the copying capabilities available, will there be a need to detect the execution environment? Does TFX rely on other tools to execute a copy that handle this? Is detection of the execution environment and the copying itself separate? What could be reused? 
    
  - If it's not easy to detect the execution environment without also performing a copy, will the user have to specify the execution environment and therefore how to do the copy (e.g., local copy, GCS, S3). And then what's the best way to handle that?
    
  2. Should the dictionary of file inputs take a path to a folder? Globs? Lists of individual files?
  3. Assuming file copying is done entirely separately, [importer.generate_output_dict](https://github.com/tensorflow/tfx/blob/f8ce19339568ae58519d4eecfdd73078f80f84a2/tfx/dsl/components/common/importer.py#L153) be used as is to register the artifacts, or does some separate code using [MLMD](https://www.tensorflow.org/tfx/guide/mlmd) in a different way need to be written


## Project Team
Alex Ho, alexanderho@google.com, @alxndrnh

