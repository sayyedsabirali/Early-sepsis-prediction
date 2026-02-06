import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME: str = "sepsis_early_warning"
PIPELINE_NAME: str = "sepsis_pipeline"
ARTIFACT_DIR: str = "artifacts"
CURRENT_YEAR = date.today().year

# ==================================================
# DATASET & TARGET
# ==================================================
TARGET_COLUMN: str = "label_12h"
RAW_DATA_FILE_NAME: str = "sepsis-data"
SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")

# ==================================================
# AWS / S3 CONFIGURATION
# ==================================================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

S3_BUCKET_NAME: str = "sepsis-mlops-data"
S3_RAW_DATA_DIR: str = "raw-data"
S3_PROCESSED_DATA_DIR: str = "processed-data"
S3_FEATURE_STORE_DIR: str = "feature-store"
S3_ARTIFACT_DIR: str = "artifacts"
S3_MODEL_REGISTRY_DIR: str = "model-registry"

# ==================================================
# DUAL MODEL CONFIGURATION
# ==================================================
# Model 1: High Recall (Screening Model)
S3_WARNING_MODEL_KEY: str = f"{S3_MODEL_REGISTRY_DIR}/warning_model.pkl"
# Model 2: High Precision (Confirmation Model)
S3_CONFIRMATION_MODEL_KEY: str = f"{S3_MODEL_REGISTRY_DIR}/confirmation_model.pkl"

# ==================================================
# DATA INGESTION
# ==================================================
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.30
DATA_INGESTION_SPLIT_BY_COLUMN: str = "stay_id"

# ==================================================
# DATA VALIDATION
# ==================================================
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "validation_report.yaml"

# ==================================================
# DATA TRANSFORMATION
# ==================================================
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMER_OBJECT_FILE: str = "transformer.pkl"

# ==================================================
# MODEL TRAINER (DUAL MODELS)
# ==================================================
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_WARNING_MODEL_NAME: str = "warning_model.pkl"
MODEL_TRAINER_CONFIRMATION_MODEL_NAME: str = "confirmation_model.pkl"

MODEL_TRAINER_WARNING_TARGET_RECALL: float = 0.73  # High recall
MODEL_TRAINER_CONFIRMATION_TARGET_PRECISION: float = 0.99  # High precision

MODEL_TRAINER_EXPECTED_AUC: float = 0.80

# ==================================================
# MODEL EVALUATION (DUAL MODELS)
# ==================================================
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02

# ==================================================
# MODEL PUSHER (DUAL MODELS)
# ==================================================
MODEL_PUSHER_BUCKET_NAME: str = S3_BUCKET_NAME
MODEL_PUSHER_WARNING_MODEL_KEY: str = S3_WARNING_MODEL_KEY
MODEL_PUSHER_CONFIRMATION_MODEL_KEY: str = S3_CONFIRMATION_MODEL_KEY

# ==================================================
# FASTAPI APPLICATION
# ==================================================
APP_HOST: str = "0.0.0.0"
APP_PORT: int = 5000

# ==================================================
# LOGGING
# ==================================================
LOG_DIR: str = "logs"

# ==================================================
# Model Parameters
# ==================================================

# Warning Model (XGBoost - High Recall)
XGBOOST_MODEL_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 12,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3,
    "eval_metric": "logloss",
    "random_state": 42,
    "use_label_encoder": False,
}

# Confirmation Model 
RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 12,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}

# ==================================================
# RISK THRESHOLDS
# ==================================================
SEPSIS_WARNING_THRESHOLD: float = 0.15  # Low threshold for warnings
SEPSIS_CONFIRMATION_THRESHOLD: float = 0.70  # High threshold for confirmation
SEPSIS_MODERATE_RISK_THRESHOLD: float = 0.40

MIN_ACCEPTABLE_RECALL: float = 0.60
MIN_ACCEPTABLE_PRECISION: float = 0.80