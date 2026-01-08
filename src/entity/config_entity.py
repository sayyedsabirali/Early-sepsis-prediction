from dataclasses import dataclass
import os
from src.constants import *


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)
    raw_data_s3_dir: str = S3_RAW_DATA_DIR
    raw_data_file_name: str = RAW_DATA_FILE_NAME
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    split_by_column: str = DATA_INGESTION_SPLIT_BY_COLUMN


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(ARTIFACT_DIR, DATA_VALIDATION_DIR_NAME)
    schema_file_path: str = SCHEMA_FILE_PATH
    report_file_path: str = os.path.join(
        ARTIFACT_DIR,
        DATA_VALIDATION_DIR_NAME,
        DATA_VALIDATION_REPORT_FILE_NAME
    )


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        ARTIFACT_DIR,
        DATA_TRANSFORMATION_DIR_NAME
    )

    transformed_train_path: str = os.path.join(
        ARTIFACT_DIR,
        DATA_TRANSFORMATION_DIR_NAME,
        "train.npy"
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


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME)
    trained_model_path: str = os.path.join(
        ARTIFACT_DIR,
        MODEL_TRAINER_DIR_NAME,
        MODEL_TRAINER_TRAINED_MODEL_NAME
    )
    model_type: str = MODEL_TRAINER_MODEL_TYPE
    expected_auc: float = MODEL_TRAINER_EXPECTED_AUC
    random_state: int = MODEL_TRAINER_RANDOM_STATE


@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    model_registry_key: str = MODEL_PUSHER_S3_KEY


@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_PUSHER_BUCKET_NAME
    model_registry_key: str = MODEL_PUSHER_S3_KEY

