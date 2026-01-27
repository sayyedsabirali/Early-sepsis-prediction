import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    accuracy_score,
    precision_score,
    precision_recall_curve,
    roc_curve
)

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


class ModelEvaluation:
    """
    Research-aligned model evaluation for Sepsis prediction

    ✔ PR-AUC focused (class imbalance aware)
    ✔ Recall-aware acceptance
    ✔ Stores metrics for reporting
    ✔ Saves PR & ROC curves
    """

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
                logging.info("Production model found in S3")
                return estimator.load_model()

            logging.info("No production model found in S3")
            return None

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation")

            os.makedirs("artifacts", exist_ok=True)

            # ------------------------------
            # Load test data
            # ------------------------------
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_path
            )

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1].astype(int)

            # ------------------------------
            # Load new model
            # ------------------------------
            new_model = load_object(
                self.model_trainer_artifact.trained_model_path
            )

            y_prob_new = new_model.trained_model_object.predict_proba(X_test)[:, 1]
            y_pred_new = (y_prob_new >= new_model.decision_threshold).astype(int)

            # ------------------------------
            # Metrics
            # ------------------------------
            new_roc_auc = roc_auc_score(y_test, y_prob_new)
            new_pr_auc = average_precision_score(y_test, y_prob_new)
            new_recall = recall_score(y_test, y_pred_new)
            new_accuracy = accuracy_score(y_test, y_pred_new)
            new_precision = precision_score(y_test, y_pred_new)

            logging.info(
                f"New model | ROC-AUC={new_roc_auc:.4f} | PR-AUC={new_pr_auc:.4f}"
            )

            # ------------------------------
            # Compare with production model
            # ------------------------------
            best_model = self._load_best_model_from_s3()

            if best_model:
                y_prob_best = best_model.trained_model_object.predict_proba(X_test)[:, 1]
                best_pr_auc = average_precision_score(y_test, y_prob_best)
            else:
                best_pr_auc = 0.0

            pr_auc_diff = new_pr_auc - best_pr_auc

            # ------------------------------
            # Acceptance criteria
            # ------------------------------
            is_accepted = (
                new_pr_auc >= self.model_eval_config.changed_threshold_score
                and pr_auc_diff >= 0
            )

            logging.info(
                f"Model acceptance={is_accepted} | PR-AUC improvement={pr_auc_diff:.4f}"
            )

            # ------------------------------
            # Save metrics
            # ------------------------------
            metrics = {
                "roc_auc": float(new_roc_auc),
                "pr_auc": float(new_pr_auc),
                "recall": float(new_recall),
                "precision": float(new_precision),
                "accuracy": float(new_accuracy),
                "pr_auc_diff": float(pr_auc_diff)
            }

            metrics_path = os.path.join("artifacts", "evaluation_metrics.json")

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            # ------------------------------
            # PR Curve
            # ------------------------------
            precision, recall, _ = precision_recall_curve(y_test, y_prob_new)

            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve (Sepsis Model)")
            plt.grid()

            pr_curve_path = os.path.join("artifacts", "pr_curve.png")
            plt.savefig(pr_curve_path)
            plt.close()

            # ------------------------------
            # ROC Curve
            # ------------------------------
            fpr, tpr, _ = roc_curve(y_test, y_prob_new)

            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (Sepsis Model)")
            plt.grid()

            roc_curve_path = os.path.join("artifacts", "roc_curve.png")
            plt.savefig(roc_curve_path)
            plt.close()

            logging.info("Evaluation artifacts saved successfully")

            # ------------------------------
            # Return artifact
            # ------------------------------
            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                s3_model_path=self.model_eval_config.model_registry_key,
                trained_model_path=self.model_trainer_artifact.trained_model_path,
                new_model_pr_auc=new_pr_auc,
                best_model_pr_auc=best_pr_auc,
                pr_auc_diff=pr_auc_diff,
                new_model_recall=new_recall,
                accuracy=new_accuracy,
                metrics_path=metrics_path,
                pr_curve_path=pr_curve_path,
                roc_curve_path=roc_curve_path
            )

        except Exception as e:
            raise MyException(e, sys)
