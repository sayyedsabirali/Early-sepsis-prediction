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



@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    s3_model_path: str
    trained_model_path: str

    # Core research metrics
    new_model_pr_auc: float
    best_model_pr_auc: float
    pr_auc_diff: float
    new_model_recall: float

    # 🔥 Newly added reporting fields
    accuracy: float
    metrics_path: str
    pr_curve_path: str
    roc_curve_path: str


# ==============================
# Model Pusher
# ==============================
@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
    message: str
