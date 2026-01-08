import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import roc_auc_score, average_precision_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)
from src.entity.s3_estimator import S3ModelEstimator
from src.utils.main_utils import load_numpy_array_data, load_object
from src.exception import MyException
from src.logger import logging


@dataclass
class EvaluateModelResponse:
    new_model_roc_auc: float
    new_model_pr_auc: float
    best_model_roc_auc: float
    is_model_accepted: bool
    roc_auc_diff: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.model_eval_config = model_eval_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def _load_best_model_from_s3(self):
        try:
            estimator = S3ModelEstimator(
                bucket_name=self.model_eval_config.bucket_name,
                model_path=self.model_eval_config.model_registry_key,
            )

            if estimator.is_model_present():
                try:
                    # 🔑 IMPORTANT: validate model integrity
                    estimator.load_model()
                    logging.info("Valid production model found in S3")
                    return estimator
                except Exception:
                    logging.warning(
                        "Model registry path exists but model file is missing or corrupt. "
                        "Ignoring production model."
                    )
                    return None

            logging.info("No existing production model found in S3")
            return None

        except Exception as e:
            raise MyException(e, sys)



    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        test_arr = load_numpy_array_data(
            self.data_transformation_artifact.transformed_test_path
        )
        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1].astype(int)

        new_model = load_object(self.model_trainer_artifact.trained_model_path)
        y_prob_new = new_model.trained_model_object.predict_proba(X_test)[:, 1]

        new_roc_auc = roc_auc_score(y_test, y_prob_new)

        best_model_estimator = self._load_best_model_from_s3()
        best_roc_auc = 0.0

        if best_model_estimator:
            best_model = best_model_estimator.load_model()
            y_prob_best = best_model.trained_model_object.predict_proba(X_test)[:, 1]
            best_roc_auc = roc_auc_score(y_test, y_prob_best)

        is_accepted = (
            new_roc_auc >= self.model_eval_config.changed_threshold_score
            and new_roc_auc > best_roc_auc
        )

        return ModelEvaluationArtifact(
            is_model_accepted=is_accepted,
            s3_model_path=self.model_eval_config.model_registry_key,
            trained_model_path=self.model_trainer_artifact.trained_model_path,
            changed_accuracy=new_roc_auc - best_roc_auc,
        )
