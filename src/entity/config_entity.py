from dataclasses import dataclass
import os
from src.constants import *


# ==================================================
# DATA INGESTION
# ==================================================
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)
    raw_data_s3_dir: str = S3_RAW_DATA_DIR
    raw_data_file_name: str = RAW_DATA_FILE_NAME
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    split_by_column: str = DATA_INGESTION_SPLIT_BY_COLUMN


# ==================================================
# DATA VALIDATION
# ==================================================
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(ARTIFACT_DIR, DATA_VALIDATION_DIR_NAME)
    schema_file_path: str = SCHEMA_FILE_PATH
    report_file_path: str = os.path.join(
        ARTIFACT_DIR,
        DATA_VALIDATION_DIR_NAME,
        DATA_VALIDATION_REPORT_FILE_NAME
    )


# ==================================================
# DATA TRANSFORMATION
# ==================================================
@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME
    )

    transformed_train_path: str = os.path.join(
        ARTIFACT_DIR,
        DATA_TRANSFORMATION_DIR_NAME,
        "train.npy"
    )

    transformed_val_path: str = os.path.join(
        ARTIFACT_DIR,
        DATA_TRANSFORMATION_DIR_NAME,
        "val.npy"
    )

    transformed_test_path: str = os.path.join(
        ARTIFACT_DIR,
        DATA_TRANSFORMATION_DIR_NAME,
        "test.npy"
    )

    transformer_object_path: str = os.path.join(
        ARTIFACT_DIR,
        DATA_TRANSFORMATION_DIR_NAME,
        DATA_TRANSFORMATION_TRANSFORMER_OBJECT_FILE
    )


# ==================================================
# MODEL TRAINER (DUAL MODELS)
# ==================================================
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME)
    trained_model_warning_path: str = os.path.join(
        ARTIFACT_DIR,
        MODEL_TRAINER_DIR_NAME,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_TRAINER_WARNING_MODEL_NAME
    )
    trained_model_confirmation_path: str = os.path.join(
        ARTIFACT_DIR,
        MODEL_TRAINER_DIR_NAME,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_TRAINER_CONFIRMATION_MODEL_NAME
    )
    expected_auc: float = MODEL_TRAINER_EXPECTED_AUC
    trained_model_dir: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME, MODEL_TRAINER_TRAINED_MODEL_DIR)  # ADD THIS LINE


# ==================================================
# MODEL EVALUATION (DUAL MODELS)
# ==================================================
@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    warning_model_registry_key: str = MODEL_PUSHER_WARNING_MODEL_KEY
    confirmation_model_registry_key: str = MODEL_PUSHER_CONFIRMATION_MODEL_KEY


# ==================================================
# MODEL PUSHER (DUAL MODELS)
# ==================================================
@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    warning_model_registry_key: str = MODEL_PUSHER_WARNING_MODEL_KEY
    confirmation_model_registry_key: str = MODEL_PUSHER_CONFIRMATION_MODEL_KEY