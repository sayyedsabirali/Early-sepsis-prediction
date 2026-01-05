import os
import sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logger
from src.data_access.sepsis_data import SepsisData


class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.sepsis_data = SepsisData()
        except Exception as e:
            raise MyException(e, sys)

    def export_data_from_s3(self) -> DataFrame:
        try:
            logger.info("Fetching raw data from S3")
            df = self.sepsis_data.export_s3_data_as_dataframe()
            logger.info(f"Raw data shape: {df.shape}")
            return df
        except Exception as e:
            raise MyException(e, sys)

    def split_data_train_val_test(self, dataframe: DataFrame):
        try:
            logger.info("Splitting data into train / val / test")

            # 70% train, 30% temp
            train_df, temp_df = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                shuffle=True
            )

            # Split remaining 30% into 15% val, 15% test
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                random_state=42,
                shuffle=True
            )

            ingestion_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(ingestion_dir, exist_ok=True)

            train_path = os.path.join(ingestion_dir, "train.csv")
            val_path = os.path.join(ingestion_dir, "val.csv")
            test_path = os.path.join(ingestion_dir, "test.csv")

            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info("Train / Val / Test split completed")

            return train_path, val_path, test_path

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion")

            df = self.export_data_from_s3()

            train_path, val_path, test_path = self.split_data_train_val_test(df)

            artifact = DataIngestionArtifact(
                train_file_path=train_path,
                test_file_path=test_path,
                is_ingested=True,
                message="Train/Val/Test data ingestion completed successfully"
            )

            logger.info(f"DataIngestionArtifact: {artifact}")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
