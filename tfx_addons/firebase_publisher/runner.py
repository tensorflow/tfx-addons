from typing import Any, Dict, List, Optional
from absl import logging

import os
import glob
import tempfile
from tfx.utils import io_utils

import firebase_admin
from firebase_admin import ml
from firebase_admin import credentials


def deploy_model_for_firebase_ml(
    display_name: str,
    storage_bucket: str,
    tags: List[Any],
    options: Dict[str, Any],
    model_path: str,
    model_version: str,
    credential_path: Optional[str],
) -> str:
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
    tags.append(model_version)

    firebase_admin.initialize_app(credential=credential, options=options)
    logging.info("firebase app initialization is completed")
    
    tmp_model_path = os.path.join(tmp_dir, "model")
    io_utils.copy_dir(model_path, tmp_model_path)
    
    tflite_files = glob.glob(f"{tmp_model_path}/**/*.tflite")
    is_tflite = len(tflite_files) > 0
    model_path = tflite_files[0] if is_tflite else tmp_model_path
    
    if is_tflite:
        source = ml.TFLiteGCSModelSource.from_tflite_model_file(model_path)
    else:
        source = ml.TFLiteGCSModelSource.from_saved_model(model_path)

    model_list = ml.list_models(list_filter=f"display_name={display_name}")
    
    # update existing model
    if len(model_list.models) > 0:
        # get the first match model
        model = model_list.models[0]
        model.tags = tags
        model.model_format = ml.TFLiteFormat(model_source=source)

        updated_model = ml.update_model(model)
        ml.publish_model(updated_model.model_id)

        logging.info("model exists, so it is updated")
    # create a new model
    else:
        # create the model object
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

    return ""
