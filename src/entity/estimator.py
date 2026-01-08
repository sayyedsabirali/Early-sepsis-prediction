import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging
from src.utils.preprocessing_utils import PreprocessingUtils


class MyModel:
    """
    Wrapper class containing:
    - preprocessing pipeline
    - trained ML model
    Used during inference
    """

    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame):
        try:
            logging.info("Starting inference pipeline")

            # Step 1: preprocessing (same as training)
            df_processed = PreprocessingUtils.apply_preprocessing_transformations(
                dataframe
            )

            # Step 2: apply transformer 
            X = self.preprocessing_object.transform(df_processed)

            # Step 3: prediction
            preds = self.trained_model_object.predict(X)

            logging.info(f"Inference completed | samples={len(preds)}")
            return preds

        except Exception as e:
            raise MyException(e, sys)
