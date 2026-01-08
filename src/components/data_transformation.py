import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from src.constants import TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import save_object, save_numpy_array_data
from src.utils.preprocessing_utils import PreprocessingUtils


def identity_function(x):
    return x


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

    def get_transformer(self) -> Pipeline:
        logger.info("Creating identity transformer (NO scaling)")
        return Pipeline(
            steps=[
                (
                    "identity",
                    ColumnTransformer(
                        transformers=[
                            ("identity", FunctionTransformer(identity_function), slice(0, None))
                        ],
                        remainder="passthrough",
                    ),
                )
            ]
        )

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting Data Transformation")

            if not self.validation_artifact.validation_status:
                raise Exception(self.validation_artifact.message)

            # Load data
            train_df = self.read_data(self.ingestion_artifact.train_file_path)
            val_df = self.read_data(self.ingestion_artifact.val_file_path)
            test_df = self.read_data(self.ingestion_artifact.test_file_path)

            # Feature engineering (skipped by design)
            train_df = PreprocessingUtils.apply_complete_feature_engineering(train_df)
            val_df   = PreprocessingUtils.apply_complete_feature_engineering(val_df)
            test_df  = PreprocessingUtils.apply_complete_feature_engineering(test_df)

            # Separate X/y
            y_train = train_df[TARGET_COLUMN]
            y_val   = val_df[TARGET_COLUMN]
            y_test  = test_df[TARGET_COLUMN]

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            X_val   = val_df.drop(columns=[TARGET_COLUMN])
            X_test  = test_df.drop(columns=[TARGET_COLUMN])

            # Preprocessing
            X_train = PreprocessingUtils.apply_preprocessing_transformations(X_train)
            X_val   = PreprocessingUtils.apply_preprocessing_transformations(X_val)
            X_test  = PreprocessingUtils.apply_preprocessing_transformations(X_test)

            # Transformer
            transformer = self.get_transformer()

            X_train_arr = transformer.fit_transform(X_train)
            X_val_arr   = transformer.transform(X_val)
            X_test_arr  = transformer.transform(X_test)

            train_arr = np.c_[X_train_arr, y_train.values]
            val_arr   = np.c_[X_val_arr, y_val.values]
            test_arr  = np.c_[X_test_arr, y_test.values]

            # Save
            save_numpy_array_data(self.config.transformed_train_path, train_arr)
            save_numpy_array_data(self.config.transformed_test_path, test_arr)
            save_object(self.config.transformer_object_path, transformer)

            logger.info("Data Transformation completed successfully")

            return DataTransformationArtifact(
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
                transformer_object_path=self.config.transformer_object_path,
                message="Data transformation successful",
            )

        except Exception as e:
            raise MyException(e, sys)
