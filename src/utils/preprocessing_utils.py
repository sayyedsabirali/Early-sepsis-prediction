import sys
import numpy as np
import pandas as pd
from src.exception import MyException
from src.logger import logger


class PreprocessingUtils:

    @staticmethod
    def apply_complete_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        Currently skipping feature engineering as per requirements.
        """
        try:
            logger.info("Skipping feature engineering as per configuration")
            return df.copy()
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def apply_preprocessing_transformations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete preprocessing pipeline:
        1. Drop unnecessary columns
        2. Map gender to numeric values
        3. Handle missing values
        4. Label encode categorical columns
        """
        try:
            df = df.copy()
            
            # Step 1: Drop unnecessary columns
            df = PreprocessingUtils._drop_columns(df)
            
            # Step 2: Map gender to numeric values
            df = PreprocessingUtils._map_gender(df)
            
            # Step 3: Handle missing values
            df = PreprocessingUtils._handle_missing_values(df)
            
            # Step 4: Label encode categorical columns
            df = PreprocessingUtils._label_encode_columns(df)
            
            # Final cleanup: Handle infinities and remaining NaN values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            
            logger.info(f"Preprocessing completed | Final shape: {df.shape}")
            return df

        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns that are not required for modeling
        """
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
        
        columns_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
        if columns_to_drop:
            logger.info(f"Dropping columns: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop, errors="ignore")
        
        return df

    @staticmethod
    def _map_gender(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map gender values to numeric codes:
        F -> 0, M -> 1, Others/Unknown -> 2
        """
        if "gender" in df.columns:
            df["gender"] = (
                df["gender"]
                .map({"F": 0, "M": 1})
                .fillna(2)
                .astype(int)
            )
            logger.info("Gender column mapped to numeric values")
        return df

    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with mixed strategy:
        - Forward-fill for time-dependent clinical signals
        - Median-fill for standard numeric features
        - 'unknown' for categorical features
        """
        
        # Columns that should be forward-filled (time series data)
        FFILL_COLS = ["urine_output_6h_ml"]  # Only columns that are NOT in DROP_COLUMNS
        
        for col in FFILL_COLS:
            if col in df.columns:
                df[col] = df[col].ffill()
                logger.info(f"Forward-filled missing values for: {col}")
        
        # Handle remaining columns
        for col in df.columns:
            if df[col].dtype == "object":
                # Categorical columns
                df[col] = df[col].fillna("unknown")
            elif col not in FFILL_COLS:
                # Numeric columns (excluding already forward-filled columns)
                median_val = df[col].median()
                if pd.isna(median_val) or np.isinf(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
        
        return df

    @staticmethod
    def _label_encode_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Label encode remaining categorical columns
        """
        CATEGORICAL_COLS = []
        
        # Check for any remaining categorical columns (excluding already processed ones)
        for col in df.columns:
            if df[col].dtype == "object" and col != "gender":
                CATEGORICAL_COLS.append(col)
        
        if CATEGORICAL_COLS:
            logger.info(f"Label encoding columns: {CATEGORICAL_COLS}")
        
        for col in CATEGORICAL_COLS:
            df[col] = pd.factorize(df[col])[0]
            logger.info(f"Encoded column: {col}")
        
        return df

    @staticmethod
    def get_required_columns() -> list:
        """
        Return list of columns required for model input
        """
        REQUIRED_COLUMNS = [
            'hr',
            'map', 
            'rr',
            'temp_c',
            'spo2',
            'creatinine',
            'bilirubin',
            'platelets',
            'lactate',
            'urine_output_6h_ml',
            'anchor_age',
            'gender'
        ]
        
        return REQUIRED_COLUMNS

    @staticmethod
    def validate_input_data(df: pd.DataFrame) -> bool:
        """
        Validate that input dataframe has all required columns
        """
        try:
            required_cols = PreprocessingUtils.get_required_columns()
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            logger.info("Input data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating input data: {e}")
            return False

    @staticmethod
    def ensure_column_order(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure dataframe columns are in correct order for model input
        """
        try:
            required_cols = PreprocessingUtils.get_required_columns()
            
            # Add any extra columns that might be present
            current_cols = list(df.columns)
            extra_cols = [col for col in current_cols if col not in required_cols]
            
            # Order: required columns first, then any extra columns
            ordered_cols = required_cols + extra_cols
            
            # Reorder dataframe
            df = df[ordered_cols]
            
            return df
            
        except Exception as e:
            logger.warning(f"Could not reorder columns: {e}")
            return df