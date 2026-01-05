import sys
import os
import json
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def validate_columns(self, df: DataFrame) -> list:
        expected_cols = set(self.schema["columns"])
        actual_cols = set(df.columns)
        missing_cols = list(expected_cols - actual_cols)
        return missing_cols

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting Data Validation")

            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            validation_errors = []

            # Empty check
            if train_df.empty or test_df.empty:
                validation_errors.append("Train or Test dataset is empty")

            # Column check
            train_missing = self.validate_columns(train_df)
            test_missing = self.validate_columns(test_df)

            if train_missing:
                validation_errors.append(f"Missing columns in train data: {train_missing}")

            if test_missing:
                validation_errors.append(f"Missing columns in test data: {test_missing}")

            # Target check
            if "label_12h" not in train_df.columns:
                validation_errors.append("Target column label_12h missing in train data")

            if "label_12h" not in test_df.columns:
                validation_errors.append("Target column label_12h missing in test data")

            validation_status = len(validation_errors) == 0

            # Save validation report
            os.makedirs(
                os.path.dirname(self.data_validation_config.report_file_path),
                exist_ok=True
            )

            report = {
                "validation_status": validation_status,
                "errors": validation_errors
            }

            with open(self.data_validation_config.report_file_path, "w") as f:
                json.dump(report, f, indent=4)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                report_file_path=self.data_validation_config.report_file_path,
                message="; ".join(validation_errors)
            )

            logger.info(f"Data Validation completed with status: {validation_status}")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
