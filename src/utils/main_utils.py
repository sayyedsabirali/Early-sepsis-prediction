import os
import sys
import boto3
import joblib
import numpy as np
import dill
import yaml
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise MyException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise MyException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array, allow_pickle=True) 
    except Exception as e:
        raise MyException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj, allow_pickle=True) 
    except Exception as e:
        raise MyException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    Save object using joblib with compression
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 🔥 Compression level 3 = good balance
        joblib.dump(obj, file_path, compress=("lzma", 3))

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logging.info(f"Model saved at {file_path} | Size: {file_size_mb:.2f} MB")

    except Exception as e:
        raise MyException(e, sys) from e

def load_object(file_path: str) -> object:
    """
    Load object using joblib.
    If local file not found, download from S3 automatically.
    """
    try:
        # If file exists locally
        if os.path.exists(file_path):
            return joblib.load(file_path)

        logging.info(f"Local file not found: {file_path}")
        logging.info("Attempting to download from S3...")

        bucket = os.getenv("S3_BUCKET")
        s3_key_prefix = os.getenv("S3_MODEL_PREFIX", "")

        if bucket is None:
            raise Exception("S3_BUCKET environment variable not set")

        filename = os.path.basename(file_path)
        s3_key = f"{s3_key_prefix}/{filename}" if s3_key_prefix else filename

        s3 = boto3.client("s3")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        s3.download_file(bucket, s3_key, file_path)

        logging.info(f"Downloaded {s3_key} from S3")

        return joblib.load(file_path)

    except Exception as e:
        raise MyException(e, sys) from e
