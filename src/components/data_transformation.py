import sys
import numpy as np
import pandas as pd

from src.constants import TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import save_numpy_array_data
from src.utils.preprocessing_utils import PreprocessingUtils


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.ingestion_artifact = data_ingestion_artifact
            self.config = data_transformation_config
            self.validation_artifact = data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting Data Transformation")

            if not self.validation_artifact.validation_status:
                raise Exception(self.validation_artifact.message)

            # ============================
            # Load data
            # ============================
            train_df = self.read_data(self.ingestion_artifact.train_file_path)
            val_df   = self.read_data(self.ingestion_artifact.val_file_path)
            test_df  = self.read_data(self.ingestion_artifact.test_file_path)

            # ============================
            # Feature engineering (NO LEAK)
            # ============================
            train_df = PreprocessingUtils.apply_complete_feature_engineering(train_df)
            val_df   = PreprocessingUtils.apply_complete_feature_engineering(val_df)
            test_df  = PreprocessingUtils.apply_complete_feature_engineering(test_df)

            # ============================
            # Separate X / y
            # ============================
            y_train = train_df[TARGET_COLUMN]
            y_val   = val_df[TARGET_COLUMN]
            y_test  = test_df[TARGET_COLUMN]

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            X_val   = val_df.drop(columns=[TARGET_COLUMN])
            X_test  = test_df.drop(columns=[TARGET_COLUMN])

            # ============================
            # Label distribution check
            # ============================
            logger.info("Label distribution check:")
            logger.info(f"Train:\n{y_train.value_counts(normalize=True)}")
            logger.info(f"Val:\n{y_val.value_counts(normalize=True)}")
            logger.info(f"Test:\n{y_test.value_counts(normalize=True)}")

            # ============================
            # Preprocessing (SAFE)
            # ============================
            X_train = PreprocessingUtils.apply_preprocessing_transformations(X_train)
            X_val   = PreprocessingUtils.apply_preprocessing_transformations(X_val)
            X_test  = PreprocessingUtils.apply_preprocessing_transformations(X_test)

            # ============================
            # Convert to numpy
            # ============================
            train_arr = np.c_[X_train.values, y_train.values]
            val_arr   = np.c_[X_val.values, y_val.values]
            test_arr  = np.c_[X_test.values, y_test.values]

            # ============================
            # Save arrays
            # ============================
            save_numpy_array_data(self.config.transformed_train_path, train_arr)
            save_numpy_array_data(self.config.transformed_val_path, val_arr)
            save_numpy_array_data(self.config.transformed_test_path, test_arr)

            logger.info("Data Transformation completed successfully")

            return DataTransformationArtifact(
                transformed_train_path=self.config.transformed_train_path,
                transformed_val_path=self.config.transformed_val_path,
                transformed_test_path=self.config.transformed_test_path,
                transformer_object_path=None,  # intentionally None
                message="Data transformation successful",
            )

        except Exception as e:
            raise MyException(e, sys)
