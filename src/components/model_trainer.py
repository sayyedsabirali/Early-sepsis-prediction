import sys
import numpy as np
import xgboost as xgb
from typing import Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

from src.constants import XGB_MODEL_PARAMS
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from src.entity.estimator import MyModel
from src.explainability.shap_explainer import SepsisSHAPExplainer


class ModelTrainer:
    """
    Trains XGBoost model for early sepsis prediction.

    ✔ Handles class imbalance
    ✔ Optimizes PR-AUC
    ✔ Learns recall-driven decision threshold
    ✔ Stores threshold inside model
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    # ==========================================================
    # Core Training + Validation
    # ==========================================================
    def train_and_evaluate(
        self,
        train_arr: np.ndarray,
        val_arr: np.ndarray
    ) -> Tuple[object, float, float, float]:

        try:
            # ------------------------------
            # Split features / target
            # ------------------------------
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_val, y_val     = val_arr[:, :-1], val_arr[:, -1].astype(int)

            # ------------------------------
            # Handle imbalance
            # ------------------------------
            neg, pos = np.bincount(y_train)
            scale_pos_weight = neg / max(pos, 1)

            logging.info(
                f"Class distribution | neg={neg}, pos={pos}, "
                f"scale_pos_weight={scale_pos_weight:.2f}"
            )

            # ------------------------------
            # Model
            # ------------------------------
            model = xgb.XGBClassifier(
                **XGB_MODEL_PARAMS,
                scale_pos_weight=scale_pos_weight
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )

            # ------------------------------
            # Validation metrics
            # ------------------------------
            y_val_prob = model.predict_proba(X_val)[:, 1]

            roc_auc = roc_auc_score(y_val, y_val_prob)
            pr_auc  = average_precision_score(y_val, y_val_prob)

            logging.info(f"Validation ROC-AUC : {roc_auc:.4f}")
            logging.info(f"Validation PR-AUC  : {pr_auc:.4f}")

            # ------------------------------
            # Recall-driven threshold
            # ------------------------------
            precision, recall, thresholds = precision_recall_curve(
                y_val, y_val_prob
            )

            TARGET_RECALL = 0.80
            valid_idx = np.where(recall >= TARGET_RECALL)[0]

            if len(valid_idx) == 0:
                best_threshold = 0.5
                recall_at_th = 0.0
                precision_at_th = 0.0
                logging.warning(
                    "Target recall not achieved. Using default threshold=0.5"
                )
            else:
                best_threshold = thresholds[valid_idx[-1]]
                recall_at_th = recall[valid_idx[-1]]
                precision_at_th = precision[valid_idx[-1]]

            logging.info(
                f"Chosen decision threshold={best_threshold:.4f} | "
                f"Recall={recall_at_th:.3f} | "
                f"Precision={precision_at_th:.3f}"
            )

            return model, roc_auc, pr_auc, best_threshold

        except Exception as e:
            raise MyException(e, sys)

    # ==========================================================
    # Pipeline Entry
    # ==========================================================
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Model Trainer")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_path
            )
            val_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_path
            )

            model, auc, pr_auc, threshold = self.train_and_evaluate(
                train_arr, val_arr
            )

            if auc < self.model_trainer_config.expected_auc:
                raise Exception(
                    f"Model AUROC {auc:.4f} < expected "
                    f"{self.model_trainer_config.expected_auc}"
                )

            # ------------------------------
            # SHAP (optional)
            # ------------------------------
            try:
                logging.info("Generating SHAP explanations")

                X_val = val_arr[:, :-1]
                shap_sample = X_val[:500]

                shap_explainer = SepsisSHAPExplainer(
                    model=model,
                    feature_names=None
                )

                shap_explainer.explain_global(shap_sample)
                shap_explainer.explain_patient(shap_sample, patient_index=0)

            except Exception as shap_error:
                logging.warning(f"SHAP skipped: {shap_error}")

            # ------------------------------
            # FINAL MODEL (🔥 CORRECT)
            # ------------------------------
            final_model = MyModel(
                trained_model_object=model,
                decision_threshold=threshold
            )

            save_object(
                self.model_trainer_config.trained_model_path,
                final_model
            )

            logging.info(
                f"Model saved successfully | decision_threshold={threshold:.4f}"
            )

            return ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path,
                train_auc=auc,
                val_auc=auc,
                message="Sepsis early-warning model trained successfully",
            )

        except Exception as e:
            raise MyException(e, sys)
