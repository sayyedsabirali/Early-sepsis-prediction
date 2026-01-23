import sys
import numpy as np
import pandas as pd
from src.exception import MyException
from src.logger import logger


class PreprocessingUtils:

    @staticmethod
    def apply_complete_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Skipping feature engineering")
            return df.copy()
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def apply_preprocessing_transformations(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df = PreprocessingUtils._drop_columns(df)
            df = PreprocessingUtils._map_gender(df)
            df = PreprocessingUtils._handle_missing_values(df)
            df = PreprocessingUtils._label_encode_columns(df)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            logger.info(f"Preprocessing completed | final shape: {df.shape}")
            return df

        except Exception as e:
            raise MyException(e, sys)
    @staticmethod
    def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
        DROP_COLUMNS = [
            "stay_id",
            "hour",
            "insurance",
            "race",
            "urine_output_ml",
            "sofa_lab",
            "sofa_cardio",
            "admission_type"

        ]

        logger.info(f"Dropping columns: {DROP_COLUMNS}")
        return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

    @staticmethod
    def _map_gender(df: pd.DataFrame) -> pd.DataFrame:
        if "gender" in df.columns:
            df["gender"] = (
                df["gender"]
                .map({"F": 0, "M": 1})
                .fillna(2)
                .astype(int)
            )
        return df

    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Mixed missing-value strategy:
        - Forward-fill for time-dependent clinical signals
        - Median-fill for standard numeric features
        - 'unknown' for categorical
        """

        # Columns that must be forward-filled
        FFILL_COLS = [
            "sofa_cardio",
            "sofa_lab",
            "urine_output_6h_ml"
        ]
        for col in FFILL_COLS:
            if col in df.columns:
                df[col] = df[col].ffill()

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("unknown")
            elif col not in FFILL_COLS:
                median = df[col].median()
                if pd.isna(median) or np.isinf(median):
                    median = 0
                df[col] = df[col].fillna(median)

        return df


    @staticmethod
    def _label_encode_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Label encode remaining categorical columns
        """
        CATEGORICAL_COLS = ["admission_type"]

        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]

        return df
