import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ==================================================
# PROJECT METADATA
# ==================================================

PROJECT_NAME: str = "sepsis_early_warning"
PIPELINE_NAME: str = "sepsis_pipeline"

ARTIFACT_DIR: str = "artifacts"

CURRENT_YEAR = date.today().year

# ==================================================
# DATASET & TARGET
# ==================================================

TARGET_COLUMN: str = "label_12h"

RAW_DATA_FILE_NAME: str = "sepsis-data" # uploaded file name in S3
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
# MODEL TRAINER
# ==================================================

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

MODEL_TRAINER_MODEL_TYPE: str = "HistGradientBoosting"
MODEL_TRAINER_RANDOM_STATE: int = 42

MODEL_TRAINER_EXPECTED_AUC: float = 0.80

# ==================================================
# MODEL EVALUATION
# ==================================================

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02

# ==================================================
# MODEL PUSHER
# ==================================================

MODEL_PUSHER_BUCKET_NAME: str = S3_BUCKET_NAME
MODEL_PUSHER_S3_KEY: str = S3_MODEL_REGISTRY_DIR

# ==================================================
# FASTAPI APPLICATION
# ==================================================

APP_HOST: str = "0.0.0.0"
APP_PORT: int = 8000

# ==================================================
# LOGGING
# ==================================================

LOG_DIR: str = "logs"