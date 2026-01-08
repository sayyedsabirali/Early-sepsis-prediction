import sys
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Tuple

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object
)
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from src.entity.estimator import MyModel

# ✅ SHAP IMPORT
from src.explainability.shap_explainer import SepsisSHAPExplainer


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def train_and_evaluate(
        self, train_arr: np.ndarray, test_arr: np.ndarray
    ) -> Tuple[object, float, float]:

        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)

            logging.info("Training XGBoost model for sepsis prediction")

            model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.7,
                gamma=0.5,
                min_child_weight=5,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False
            )

            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            logging.info(f"Validation ROC-AUC : {roc_auc:.4f}")
            logging.info(f"Validation PR-AUC  : {pr_auc:.4f}")

            return model, roc_auc, pr_auc

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Model Trainer")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_path
            )

            model, auc, pr_auc = self.train_and_evaluate(train_arr, test_arr)

            if auc < self.model_trainer_config.expected_auc:
                raise Exception(
                    f"Model AUROC {auc:.4f} < expected {self.model_trainer_config.expected_auc}"
                )

            preprocessing_obj = load_object(
                self.data_transformation_artifact.transformer_object_path
            )

            # ============================
            # ✅ SHAP EXPLAINABILITY
            # ============================
            try:
                logging.info("Generating SHAP explanations")

                X_test = test_arr[:, :-1]

                # Use only a small sample for SHAP (performance-safe)
                shap_sample = X_test[:500]

                feature_names = preprocessing_obj.feature_names_in_.tolist()

                shap_explainer = SepsisSHAPExplainer(
                    model=model,
                    feature_names=feature_names
                )

                shap_explainer.explain_global(shap_sample)
                shap_explainer.explain_patient(shap_sample, patient_index=0)

                logging.info("SHAP explainability completed")

            except Exception as shap_error:
                # SHAP failure should NEVER break training
                logging.warning(f"SHAP explainability skipped: {shap_error}")

            # ============================
            # ✅ SAVE FINAL MODEL
            # ============================
            final_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=model,
            )

            save_object(
                self.model_trainer_config.trained_model_path,
                final_model
            )

            logging.info("Model training completed and saved")

            return ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path,
                train_auc=auc,
                val_auc=auc,
                message="Sepsis model trained successfully",
            )

        except Exception as e:
            raise MyException(e, sys)
