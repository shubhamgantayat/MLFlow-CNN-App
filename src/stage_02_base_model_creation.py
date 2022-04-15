import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
from src.utils.model import log_model_summary
import tensorflow as tf


STAGE = "BASE MODEL CREATION" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    params = config['params']
    # LAYERS = [
    #     tf.keras.layers.Conv2D(32, (3, 3), input_shape=tuple(params["img_shape"]), activation="relu"),
    #     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    #     tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    #     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dense(2, activation="softmax")
    # ]

    base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(2, activation="softmax")(avg)
    classifier = tf.keras.Model(inputs=base_model.input, outputs=output)
    for layer in base_model.layers:
        layer.trainable = False


    # classifier = tf.keras.Sequential(LAYERS)
    logging.info(f"base model summary : \n {log_model_summary(classifier)}")
    optimizer = tf.keras.optimizers.SGD(learning_rate=params["lr"], momentum=0.9, decay=0.01)
    classifier.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    path_to_model_dir = os.path.join(config["data"]["local_dir"], config["data"]["model_dir"])
    create_directories([path_to_model_dir])
    path_to_model = os.path.join(path_to_model_dir, config["data"]["init_model_file"])
    classifier.save(path_to_model)
    logging.info(f"model is saved at : {path_to_model}")


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