import logging

from PIL import Image
import shutil
import imghdr
import os
from src.utils.common import create_directories


def validating_image(config: dict) -> None:
    PARENT_DIR = os.path.join(config["data"]["unzip_data_dir"],
                              config["data"]["parent_data_dir"])
    BAD_DATA_DIR = os.path.join(config["data"]["unzip_data_dir"],
                                config["data"]["bad_data_dir"])
    create_directories([BAD_DATA_DIR])
    for dirs in os.listdir(PARENT_DIR):
        full_data_path_dir = os.path.join(PARENT_DIR, dirs)
        for imgs in os.listdir(full_data_path_dir):
            path_to_img = os.path.join(full_data_path_dir, imgs)
            bad_data_path = os.path.join(BAD_DATA_DIR, imgs)
            try:
                img = Image.open(path_to_img)
                img.verify()
                if imghdr.what(path_to_img) not in ["jpeg", "png", "jpg"]:
                    shutil.move(path_to_img, bad_data_path)
                    logging.info(f"{path_to_img} is bad")
            except Exception as e:
                shutil.move(path_to_img, bad_data_path)
                logging.info(f"{path_to_img} is bad")