from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    val_file_path: str
    test_file_path: str
    is_ingested: bool
    message: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    report_file_path: str
    message: str


@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path: str
    transformer_object_path: str
    message: str


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    train_auc: float
    val_auc: float
    message: str


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    s3_model_path: str
    trained_model_path: str
    changed_accuracy: float


@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
    message: str
