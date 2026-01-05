import os
import sys
import pickle
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
    Save object using pickle with proper error handling
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)  # Use pickle instead of dill
    except Exception as e:
        raise MyException(e, sys) from e

def load_object(file_path: str) -> object:
    """
    Load object using pickle with proper error handling
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)  # Use pickle instead of dill
    except Exception as e:
        raise MyException(e, sys) from e