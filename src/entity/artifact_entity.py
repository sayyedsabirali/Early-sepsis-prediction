from dataclasses import dataclass
from typing import Optional,Dict


# ==============================
# Data Ingestion
# ==============================
@dataclass
class DataIngestionArtifact:
    train_file_path: str
    val_file_path: str
    test_file_path: str
    is_ingested: bool
    message: str


# ==============================
# Data Validation
# ==============================
@dataclass
class DataValidationArtifact:
    validation_status: bool
    report_file_path: str
    message: str


# ==============================
# Data Transformation
# ==============================
@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_val_path: str
    transformed_test_path: str
    transformer_object_path: Optional[str]
    message: str


# ==============================
# Model Trainer (DUAL MODELS)
# ==============================
@dataclass
class ModelTrainerArtifact:
    trained_warning_model_path: str
    trained_confirmation_model_path: str
    warning_model_auc: float
    confirmation_model_auc: float
    message: str


# ==============================
# Model Evaluation (DUAL MODELS)
# ==============================
@dataclass
class ModelEvaluationArtifact:
    is_warning_model_accepted: bool
    is_confirmation_model_accepted: bool
    warning_model_pr_auc: float
    confirmation_model_pr_auc: float
    warning_model_recall: float
    confirmation_model_precision: float
    metrics_path: str
    message: str
    model_training_config: Optional[Dict] = None 

# ==============================
# Model Pusher (DUAL MODELS)
# ==============================
@dataclass
class ModelPusherArtifact:
    bucket_name: str
    warning_s3_model_path: str
    confirmation_s3_model_path: str
    message: str