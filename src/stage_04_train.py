import argparse
import os
import shutil

import mlflow
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf


STAGE = "TRAINING" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    # read config files
    config = read_yaml(config_path)
    params = config["params"]

    # get the data ready
    PARENT_DIR = os.path.join(config["data"]["unzip_data_dir"],
                              config["data"]["parent_data_dir"])
    BAD_DATA_DIR = os.path.join(config["data"]["unzip_data_dir"],
                                config["data"]["bad_data_dir"])

    logging.info(f"read the data from {PARENT_DIR}")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PARENT_DIR,
        labels="inferred",
        label_mode="int",
        validation_split=params["validation_split"],
        subset="training",
        seed=params["seed"],
        image_size=params["image_size"],
        batch_size=params["batch_size"]
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PARENT_DIR,
        labels="inferred",
        label_mode="int",
        validation_split=params["validation_split"],
        subset="validation",
        seed=params["seed"],
        image_size=params["image_size"],
        batch_size=params["batch_size"]
    )

    train_ds = train_ds.prefetch(buffer_size=params["buffer_size"])
    val_ds = val_ds.prefetch(buffer_size=params["buffer_size"])

    # load the base model
    path_to_model_dir = os.path.join(config["data"]["local_dir"], config["data"]["model_dir"])
    path_to_model = os.path.join(path_to_model_dir, config["data"]["init_model_file"])
    logging.info(f"Load the base model from {path_to_model}")
    classifier = tf.keras.models.load_model(path_to_model)

    #Early Stopping
    early_stop = tf.keras.callbacks.EarlyStopping(patience=2, verbose=3, restore_best_weights=True)

    # training
    logging.info(f"training started")
    classifier.fit(train_ds, epochs=params["epochs"], validation_data=val_ds, callbacks=[early_stop])

    # save the model
    path_to_trained_model = os.path.join(path_to_model_dir, config["data"]["trained_model_file"])
    classifier.save(path_to_trained_model)
    logging.info(f"Model saved at {path_to_trained_model}")
    with mlflow.start_run() as runs:
        mlflow.log_params(params)
        mlflow.keras.log_model(classifier, "model")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e