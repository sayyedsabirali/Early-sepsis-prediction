from dataclasses import dataclass
from typing import Optional


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
# Data Transformation (🔥 FIXED)
# ==============================
@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_val_path: str        # ✅ ADDED
    transformed_test_path: str
    transformer_object_path: Optional[str]
    message: str


# ==============================
# Model Trainer
# ==============================
@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    train_auc: float
    val_auc: float
    message: str


# ==============================
# Model Evaluation
# ==============================
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    s3_model_path: str
    trained_model_path: str

    # Research-aligned metrics
    new_model_pr_auc: float
    best_model_pr_auc: float
    pr_auc_diff: float
    new_model_recall: float


# ==============================
# Model Pusher
# ==============================
@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
    message: str
