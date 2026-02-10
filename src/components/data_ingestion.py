import os
import sys
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logger


class DataIngestion:
    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)
    def export_data_from_s3(self) -> DataFrame:
        try:
            logger.info("TEMP MODE: Loading data from local machine")

            LOCAL_DATA_PATH = r"F:\9. MAJOR PROJECT\2. Sepsis- project\sepsis-data\sepsis data"
            df = pd.read_csv(LOCAL_DATA_PATH)
            logger.info(f"Loaded LOCAL sample data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

            return df

            # ===== PROD MODE =====
            # logger.info("Fetching raw data from S3")
            # df = self.sepsis_data.export_s3_data_as_dataframe()
            # logger.info(f"Raw data shape: {df.shape}")
            # return df

        except Exception as e:
            raise MyException(e, sys)

    # ==========================================================
    # ICU STAY-LEVEL STRATIFIED SPLIT (NO DATA LEAKAGE)
    # ==========================================================
    def split_data_train_val_test(self, dataframe: DataFrame):
        try:
            logger.info("Splitting data using ICU STAY-LEVEL stratified split")

            TARGET_COL = "label_12h"
            STAY_COL = "stay_id"

            # ---------- HARD CHECK ----------
            if STAY_COL not in dataframe.columns:
                raise Exception(
                    f"{STAY_COL} not found in dataset. "
                    "ICU stay-level split is required."
                )
            stay_labels = (
                dataframe[[STAY_COL, TARGET_COL]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            logger.info(
                f"Unique ICU stays: {stay_labels.shape[0]} | "
                f"Positive rate: {stay_labels[TARGET_COL].mean():.4f}"
            )

            # --------------------------------------------------
            # 2️⃣ Train vs Temp (70 / 30)
            # --------------------------------------------------
            train_stays, temp_stays = train_test_split( 
                stay_labels,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=stay_labels[TARGET_COL] 
            )

            # --------------------------------------------------
            # 3️⃣ Validation vs Test (15 / 15)
            # --------------------------------------------------
            val_stays, test_stays = train_test_split(
                temp_stays,
                test_size=0.5,
                random_state=42,
                stratify=temp_stays[TARGET_COL]
            )

            # --------------------------------------------------
            # 4️⃣ Map back to FULL TIME-SERIES
            # --------------------------------------------------
            train_df = dataframe[dataframe[STAY_COL].isin(train_stays[STAY_COL])]
            val_df   = dataframe[dataframe[STAY_COL].isin(val_stays[STAY_COL])]
            test_df  = dataframe[dataframe[STAY_COL].isin(test_stays[STAY_COL])]

            logger.info(
                f"Train rows: {train_df.shape} | "
                f"Val rows: {val_df.shape} | "
                f"Test rows: {test_df.shape}"
            )

            # --------------------------------------------------
            # 5️⃣ Save to artifacts
            # --------------------------------------------------
            ingestion_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(ingestion_dir, exist_ok=True)

            train_path = os.path.join(ingestion_dir, "train.csv")
            val_path   = os.path.join(ingestion_dir, "val.csv")
            test_path  = os.path.join(ingestion_dir, "test.csv")

            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info("ICU stay-level Train / Val / Test split completed")

            return train_path, val_path, test_path

        except Exception as e:
            raise MyException(e, sys)

    # ==========================================================
    # PIPELINE ENTRY
    # ==========================================================
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion")

            df = self.export_data_from_s3()

            train_path, val_path, test_path = self.split_data_train_val_test(df)

            artifact = DataIngestionArtifact(
                train_file_path=train_path,
                val_file_path=val_path,
                test_file_path=test_path,
                is_ingested=True,
                message="ICU stay-level Train/Val/Test ingestion completed successfully"
            )

            logger.info(f"DataIngestionArtifact created: {artifact}")
            return artifact

        except Exception as e:
            raise MyException(e, sys)
