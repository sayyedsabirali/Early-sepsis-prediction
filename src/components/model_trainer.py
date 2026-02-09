import os
import sys
import numpy as np
import xgboost as xgb
from typing import Tuple, Dict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from xgboost import XGBClassifier

from src.constants import (
    XGBOOST_PARAMS,
    EXTRA_TREES_PARAMS,
    WARNING_TARGET_RECALL,
    CONFIRMATION_TARGET_PRECISION,
    WARNING_THRESHOLD,
    CONFIRMATION_THRESHOLD
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.estimator import MyModel


class ModelTrainer:
    """
    Trains TWO models for sepsis prediction (MLflow style):
    1. Warning Model (High Recall) - XGBoost
    2. Confirmation Model (High Precision) - ExtraTrees
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def train_warning_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Tuple[object, Dict]:
        """Train XGBoost model for high recall (MLflow style)"""
        try:
            logging.info("Training Warning Model (XGBoost - High Recall)")
            
            # Calculate scale_pos_weight like MLflow
            neg = np.sum(y_train == 0)
            pos = np.sum(y_train == 1)
            scale_pos_weight = neg / max(pos, 1)
            
            logging.info(f"Class distribution: Negative={neg}, Positive={pos}, Scale Pos Weight={scale_pos_weight:.2f}")
            
            # Use EXACT MLflow parameters
            params = {
                "n_estimators": 100,
                "random_state": 42,
                "n_jobs": -1,
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "use_label_encoder": False
            }
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Get predictions on TEST DATA (like MLflow)
            y_test_prob = model.predict_proba(X_test)[:, 1]
            y_val_prob = model.predict_proba(X_val)[:, 1]
            
            # Use threshold 0.5 (like MLflow)
            threshold = 0.5
            y_test_pred = (y_test_prob >= threshold).astype(int)
            y_val_pred = (y_val_prob >= threshold).astype(int)
            
            # Calculate ALL metrics like MLflow
            test_roc_auc = roc_auc_score(y_test, y_test_prob)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            val_roc_auc = roc_auc_score(y_val, y_val_prob)
            val_recall = recall_score(y_val, y_val_pred, zero_division=0)
            val_precision = precision_score(y_val, y_val_pred, zero_division=0)
            val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            
            # Log ALL metrics like MLflow
            logging.info(f"Warning Model Performance (TEST SET - Like MLflow):")
            logging.info(f"  Recall: {test_recall:.4f} (Target: {WARNING_TARGET_RECALL})")
            logging.info(f"  Precision: {test_precision:.4f}")
            logging.info(f"  F1-Score: {test_f1:.4f}")
            logging.info(f"  ROC-AUC: {test_roc_auc:.4f}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            
            return model, {
                "test_roc_auc": test_roc_auc,
                "test_recall": test_recall,
                "test_precision": test_precision,
                "test_f1": test_f1,
                "test_accuracy": test_accuracy,
                "val_roc_auc": val_roc_auc,
                "val_recall": val_recall,
                "val_precision": val_precision,
                "val_f1": val_f1,
                "val_accuracy": val_accuracy,
                "threshold": threshold,
                "model_type": "xgboost_warning",
                "scale_pos_weight": scale_pos_weight
            }
            
        except Exception as e:
            raise MyException(e, sys)

    def train_confirmation_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[object, Dict]:
        """Train optimized ExtraTrees model for high precision"""

        try:
            logging.info("Training Confirmation Model (Optimized ExtraTrees)")

            # 🔥 Use params from constants file
            model = ExtraTreesClassifier(**EXTRA_TREES_PARAMS)

            model.fit(X_train, y_train)

            y_test_prob = model.predict_proba(X_test)[:, 1]
            y_val_prob = model.predict_proba(X_val)[:, 1]

            threshold = 0.5
            y_test_pred = (y_test_prob >= threshold).astype(int)
            y_val_pred = (y_val_prob >= threshold).astype(int)

            test_roc_auc = roc_auc_score(y_test, y_test_prob)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            logging.info("Confirmation Model Performance (Optimized):")
            logging.info(f"  Precision: {test_precision:.4f}")
            logging.info(f"  Recall: {test_recall:.4f}")
            logging.info(f"  ROC-AUC: {test_roc_auc:.4f}")

            return model, {
                "test_roc_auc": test_roc_auc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_accuracy": test_accuracy,
                "threshold": threshold,
                "model_type": "extratrees_confirmation"
            }

        except Exception as e:
            raise MyException(e, sys)



    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Main training pipeline for dual models (MLflow style)"""
        try:
            logging.info("Starting Dual Model Training Pipeline (MLflow Style)")
            logging.info(f"Warning Target Recall: {WARNING_TARGET_RECALL}")
            logging.info(f"Confirmation Target Precision: {CONFIRMATION_TARGET_PRECISION}")
            
            # Load ALL data like MLflow
            logging.info("Loading transformed data (like MLflow)...")
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_path
            )
            val_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_val_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_path
            )
            
            # Split features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_val, y_val = val_arr[:, :-1], val_arr[:, -1].astype(int)
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)
            
            logging.info(f"Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
            logging.info(f"Validation data: {X_val.shape[0]:,} samples")
            logging.info(f"Test data: {X_test.shape[0]:,} samples (for MLflow-style evaluation)")
            
            # Train Warning Model (XGBoost) - with TEST data evaluation
            logging.info("\n" + "="*50)
            logging.info("TRAINING WARNING MODEL (XGBoost - MLflow Style)")
            logging.info("="*50)
            warning_model, warning_metrics = self.train_warning_model(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Train Confirmation Model (ExtraTrees) - with TEST data evaluation
            logging.info("\n" + "="*50)
            logging.info("="*50)
            confirmation_model, confirmation_metrics = self.train_confirmation_model(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Create model objects (use threshold 0.5 for MLflow consistency)
            warning_model_obj = MyModel(
                trained_model_object=warning_model,
                decision_threshold=0.5,  # MLflow used 0.5
                model_type="warning"
            )
            
            confirmation_model_obj = MyModel(
                trained_model_object=confirmation_model,
                decision_threshold=confirmation_metrics["threshold"],
                model_type="confirmation"
            )

            
            # Save models locally
            logging.info("Saving trained models...")
            save_object(
                self.model_trainer_config.trained_model_warning_path,
                warning_model_obj
            )
            
            save_object(
                self.model_trainer_config.trained_model_confirmation_path,
                confirmation_model_obj
            )
            
            # Create and return artifact
            artifact = ModelTrainerArtifact(
                trained_warning_model_path=self.model_trainer_config.trained_model_warning_path,
                trained_confirmation_model_path=self.model_trainer_config.trained_model_confirmation_path,
                warning_model_auc=warning_metrics["test_roc_auc"],
                confirmation_model_auc=confirmation_metrics["test_roc_auc"],
                message=f"MLflow-style dual models trained: Warning Recall={warning_metrics['test_recall']:.4f}, Confirmation Precision={confirmation_metrics['test_precision']:.4f}"
            )
            
            # Final success logging with ALL metrics like MLflow
            logging.info("\n" + "="*80)
            logging.info("MLFLOW-STYLE DUAL MODEL TRAINING COMPLETED!")
            logging.info("="*80)
            
            logging.info(" WARNING MODEL (XGBoost) - TEST SET METRICS:")
            logging.info(f"  Accuracy: {warning_metrics['test_accuracy']:.4f}")
            logging.info(f"Precision: {warning_metrics['test_precision']:.4f}")
            logging.info(f"Recall: {warning_metrics['test_recall']:.4f} (Target: {WARNING_TARGET_RECALL})")
            logging.info(f"F1-Score: {warning_metrics['test_f1']:.4f}")
            logging.info(f"ROC-AUC: {warning_metrics['test_roc_auc']:.4f}")
            
            logging.info("\nCONFIRMATION MODEL (ExtraTrees) - TEST SET METRICS:")
            logging.info(f"Accuracy: {confirmation_metrics['test_accuracy']:.4f}")
            logging.info(f"Precision: {confirmation_metrics['test_precision']:.4f} (Target: {CONFIRMATION_TARGET_PRECISION})")
            logging.info(f"Recall: {confirmation_metrics['test_recall']:.4f}")
            logging.info(f"F1-Score: {confirmation_metrics['test_f1']:.4f}")
            logging.info(f"ROC-AUC: {confirmation_metrics['test_roc_auc']:.4f}")
            
            logging.info("\nThese metrics should match MLflow results!")
            logging.info("="*80)
            
            return artifact
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise MyException(e, sys)