import sys
from pandas import DataFrame
from typing import Optional

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel
from src.logger import logging


class S3ModelEstimator:

    def __init__(self, bucket_name: str, model_path: str):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self._loaded_model = None

    def is_model_present(self) -> bool:
        return self.s3.is_object_present(
            bucket_name=self.bucket_name,
            key=self.model_path
        )

    
    def is_object_present(self, bucket_name: str, key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=key)
            return True
        except:
            return False


    def load_model(self) -> MyModel:
        """Load model from S3 (cached after first load)"""
        try:
            if self._loaded_model is None:
                logging.info("Loading model from S3...")
                self._loaded_model = self.s3.load_model(
                    model_name=self.model_path,
                    bucket_name=self.bucket_name
                )
                logging.info("Model loaded successfully from S3")
            return self._loaded_model
        except Exception as e:
            raise MyException(e, sys)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """Upload trained model to S3"""
        try:
            logging.info("Uploading trained model to S3...")
            self.s3.upload_file(
                from_filename=from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
            logging.info("Model uploaded to S3 successfully")
        except Exception as e:
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Prediction APIs
    # ------------------------------------------------------------------
    def predict(self, dataframe: DataFrame):
        """
        Predict class labels (0/1)
        """
        try:
            model = self.load_model()
            return model.predict(dataframe)
        except Exception as e:
            raise MyException(e, sys)

    def predict_proba(self, dataframe: DataFrame):
        """
        Predict probabilities (for ROC-AUC / PR-AUC / SHAP)
        """
        try:
            model = self.load_model()
            return model.predict_proba(dataframe)  # ✅ Correct!
            
        except Exception as e:
            raise MyException(e, sys)
