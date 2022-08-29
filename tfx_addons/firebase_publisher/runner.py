"""Firebase Publisher runner module.

This module handles the workflow to publish
machine learning model to Firebase ML
"""
import glob
import os
import tempfile
from typing import Any, Dict, List, Tuple

import firebase_admin
import tensorflow as tf
from absl import logging
from firebase_admin import credentials, ml
from firebase_admin.ml import ListModelsPage, TFLiteModelSource
from tfx.dsl.io import fileio
from tfx.utils import io_utils

_SIZE_LIMIT_MB = 80


def prepare_fb_download_model(app_name: str, credential_path: str,
                              storage_bucket: str, model_path: str,
                              options: Dict[str, Any]) -> str:
  """initialize Firebase app, and download the
     target model to a temporary directory"""
  tmp_dir = tempfile.gettempdir()
  credential = None

  if credential_path is not None:
    tmp_credential_path = os.path.join(tmp_dir, "credentials.json")
    io_utils.copy_file(credential_path, tmp_credential_path)
    credential = credentials.Certificate(tmp_credential_path)
    logging.info(
        "credentials are copied into a temporary directory in local filesystem"
    )

  options["storageBucket"] = storage_bucket

  firebase_admin.initialize_app(credential=credential,
                                options=options,
                                name=app_name)
  logging.info("firebase app initialization is completed")

  tmp_model_path = os.path.join(tmp_dir, "model")
  io_utils.copy_dir(model_path, tmp_model_path)

  return tmp_model_path


def get_model_path_and_type(tmp_model_path) -> Tuple[bool, str]:
  """get model path and flag if the model is TFLite"""
  tflite_files = glob.glob(f"{tmp_model_path}/**/*.tflite")
  is_tflite = len(tflite_files) > 0
  model_path = tflite_files[0] if is_tflite else tmp_model_path

  return is_tflite, model_path


def upload_model(is_tflite: bool, model_path: str) -> TFLiteModelSource:
  """upload model to GCS. If SavedModel, it will be
     converted to TFLite under the hood"""
  if is_tflite:
    source = ml.TFLiteGCSModelSource.from_tflite_model_file(model_path)
  else:
    source = ml.TFLiteGCSModelSource.from_saved_model(model_path)

  return source


def check_model_size(source: TFLiteModelSource):
  """the model size to be hosted in Firebase ML is limited
      to 40mb max. if the size exceeds, RuntimeError is raised"""
  gcs_path_for_uploaded_file = source.as_dict().get('gcsTfliteUri')
  with tf.io.gfile.GFile(gcs_path_for_uploaded_file) as f:
    file_size_in_mb = f.size() / (1 << 20)

  if file_size_in_mb > _SIZE_LIMIT_MB:
    fileio.remove(gcs_path_for_uploaded_file)
    raise RuntimeError(
        f"the file size {file_size_in_mb} exceeds"
        f"the limit of {_SIZE_LIMIT_MB}. Uploaded file is removed.")


def model_exist(model_list: ListModelsPage):
  return len(model_list.models) > 0


def update_model(model_list: ListModelsPage, source: TFLiteModelSource,
                 tags: List[str], model_version: str):
  """update existing model"""
  tags.append(model_version)

  # get the first match model
  model = model_list.models[0]

  model.tags = tags
  model.model_format = ml.TFLiteFormat(model_source=source)

  updated_model = ml.update_model(model)
  ml.publish_model(updated_model.model_id)

  logging.info("model exists, so it is updated")


def create_model(display_name: str, source: TFLiteModelSource, tags: List[str],
                 model_version: str):
  """create a new model"""
  tags.append(model_version)

  tflite_format = ml.TFLiteFormat(model_source=source)
  model = ml.Model(
      display_name=display_name,
      tags=tags,
      model_format=tflite_format,
  )

  # Add the model to your Firebase project and publish it
  new_model = ml.create_model(model)
  ml.publish_model(new_model.model_id)

  logging.info("model didn't exist, so it is created")


def deploy_model_for_firebase_ml(
    app_name: str,
    display_name: str,
    storage_bucket: str,
    tags: List[Any],
    options: Dict[str, Any],
    model_path: str,
    model_version: str,
    credential_path: str,
) -> str:
  """Executes ML model deployment workflow to Firebase ML.
       Refer to the FirebasePublisher component in component.py
       for generic description of each parameter. This docstring
       only explains how the workflow works.

       Step #1.
       workflow begins by initializing firebase app with the
       given app_name, storage_bucket, options, and credentials.
       currently, Firebase ML only let us host/upload ML model
       from the local filesystem, so it downloads a model from
       the upstream components to a temporary directory.

       Step #2.
       search the list of hosted models with display_name. this
       has to be done before uploading/hosting any ML models.
       otherwise, the number of hosted models is greater than 0,
       hence it will try to update the existing model even though
       it has to create a new model.

       Step #3.
       get the model path of the downloaded model in the local
       filesystem. along the way, it searches for any files whose
       extension is '*.tflite'. when it finds tflite model file,
       is_tflite flag is marked and the file path is obtained.
       Otherwise (in case of SavedModel), the directory of the
       SavedModel is obtained.

       step #4.
       upload/host the ML model to the designated gcs_storage bucket.
       along the way, if the given model is SavedModel, it will be
       directly converted to TFLite format before uploading

       step #5.
       check the size of the uploaded TFLite model. currently,
       Firebase ML allows us to host a model whose size is under 40mb.
       if the given TFLite model's size exceeds 40mb, then the
       uploaded model files will be deleted (rollback), and
       RuntimeError exceptioin will be raised.

       step #6.
       update the existing model. the existing tags will be replaced.
       while keeping the given tags are reserved, newly granted
       model_version will be appended to the list.

       step #7.
       create a new model.

    Returns:
        str: the GCS storage bucket path where the actual model is hosted
    """
  # Step 1
  tmp_model_path = prepare_fb_download_model(app_name, credential_path,
                                             storage_bucket, model_path,
                                             options)

  # Step 2
  model_list = ml.list_models(list_filter=f"display_name={display_name}")

  # Step 3
  is_tflite, model_path = get_model_path_and_type(tmp_model_path)

  # Step 4
  source = upload_model(is_tflite, model_path)

  # Step 5
  check_model_size(source)

  if model_exist(model_list.models):
    # Step 6
    update_model(model_list, source, tags, model_version)
  else:
    # Step 7
    create_model(display_name, source, tags, model_version)

  # Step 8
  return source.as_dict().get('gcsTfliteUri')
