import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.cloud_storage.aws_storage import SimpleStorageService
from src.constants import (
    S3_BUCKET_NAME,
    S3_RAW_DATA_DIR,
    RAW_DATA_FILE_NAME
)
from src.exception import MyException
from src.logger import logger


class SepsisData:
    """
    Data access layer for fetching Sepsis dataset from Amazon S3
    """

    def __init__(self) -> None:
        try:
            self.storage_service = SimpleStorageService()
        except Exception as e:
            raise MyException(e, sys)

    def export_s3_data_as_dataframe(
        self,
        bucket_name: str = S3_BUCKET_NAME,
        s3_dir: str = S3_RAW_DATA_DIR,
        file_name: str = RAW_DATA_FILE_NAME
    ) -> pd.DataFrame:
        """
        Fetch CSV file from S3 and return as Pandas DataFrame
        """
        try:
            logger.info("Fetching data from S3 bucket")

            s3_key = f"{s3_dir}/{file_name}"

            df = self.storage_service.read_csv(
                filename=s3_key,
                bucket_name=bucket_name
            )

            logger.info(f"Data fetched successfully with shape: {df.shape}")

            # Standard cleaning
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise MyException(e, sys)
