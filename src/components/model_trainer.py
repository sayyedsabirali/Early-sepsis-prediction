import os
import sys
import numpy as np
import xgboost as xgb
import joblib
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    recall_score,
    precision_score
)

from src.constants import (
    XGBOOST_MODEL_PARAMS,
    RANDOM_FOREST_PARAMS,
    MODEL_TRAINER_WARNING_TARGET_RECALL,
    MODEL_TRAINER_CONFIRMATION_TARGET_PRECISION
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from src.entity.estimator import MyModel


class ModelTrainer:
    """
    Trains TWO models for sepsis prediction:
    1. Warning Model (High Recall) - XGBoost
    2. Confirmation Model (High Precision) - RandomForest
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def _calculate_scale_pos_weight(self, y_train: np.ndarray) -> float:
        """Calculate class imbalance weight"""
        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / max(pos, 1)
        logging.info(
            f"Class distribution | neg={neg}, pos={pos}, "
            f"scale_pos_weight={scale_pos_weight:.2f}"
        )
        return scale_pos_weight

    def _find_warning_threshold(self, y_val: np.ndarray, y_val_prob: np.ndarray) -> float:
        """Find threshold for high recall (warning model)"""
        precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
        
        # Find threshold that gives at least target recall
        valid_idx = np.where(recall >= MODEL_TRAINER_WARNING_TARGET_RECALL)[0]
        
        if len(valid_idx) == 0:
            logging.warning("Target recall not achieved. Using threshold=0.5")
            return 0.5
        
        # Among thresholds meeting recall target, pick the one with max precision
        best_idx = valid_idx[np.argmax(precision[valid_idx])]
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        logging.info(
            f"Warning Model | Threshold={best_threshold:.4f} | "
            f"Recall={recall[best_idx]:.3f} | Precision={precision[best_idx]:.3f}"
        )
        return best_threshold

    def _find_confirmation_threshold(self, y_val: np.ndarray, y_val_prob: np.ndarray) -> float:
        """Find threshold for high precision (confirmation model)"""
        precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
        
        # Find threshold that gives at least target precision
        valid_idx = np.where(precision >= MODEL_TRAINER_CONFIRMATION_TARGET_PRECISION)[0]
        
        if len(valid_idx) == 0:
            # If can't achieve target precision, find max precision
            best_idx = np.argmax(precision[:-1])
            best_threshold = thresholds[best_idx]
            logging.warning(f"Target precision not achieved. Using best available: {precision[best_idx]:.3f}")
        else:
            # Among thresholds meeting precision target, pick the one with max recall
            best_idx = valid_idx[np.argmax(recall[valid_idx])]
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        
        logging.info(
            f"Confirmation Model | Threshold={best_threshold:.4f} | "
            f"Precision={precision[best_idx]:.3f} | Recall={recall[best_idx]:.3f}"
        )
        return best_threshold

    def train_warning_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[object, Dict]:
        """Train XGBoost model for high recall"""
        try:
            logging.info("Training Warning Model (XGBoost - High Recall)")
            
            scale_pos_weight = self._calculate_scale_pos_weight(y_train)
            params = XGBOOST_MODEL_PARAMS.copy()
            params["scale_pos_weight"] = scale_pos_weight
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Validation metrics
            y_val_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_val_prob)
            pr_auc = average_precision_score(y_val, y_val_prob)
            
            # Find threshold for high recall
            threshold = self._find_warning_threshold(y_val, y_val_prob)
            
            metrics = {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "threshold": threshold,
                "model_type": "xgboost_warning"
            }
            
            return model, metrics
            
        except Exception as e:
            raise MyException(e, sys)

    def train_confirmation_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Tuple[object, Dict]:
        """Train RandomForest model for high precision"""
        try:
            logging.info("Training Confirmation Model (RandomForest - High Precision)")
            
            model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
            model.fit(X_train, y_train)
            
            # Validation metrics
            y_val_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_val_prob)
            pr_auc = average_precision_score(y_val, y_val_prob)
            
            # Find threshold for high precision
            threshold = self._find_confirmation_threshold(y_val, y_val_prob)
            
            metrics = {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "threshold": threshold,
                "model_type": "randomforest_confirmation"
            }
            
            return model, metrics
            
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Dual Model Training")
            
            # Load data
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_path
            )
            val_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_path
            )
            
            # Split features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_val, y_val = val_arr[:, :-1], val_arr[:, -1].astype(int)
            
            # Train both models
            warning_model, warning_metrics = self.train_warning_model(
                X_train, y_train, X_val, y_val
            )
            
            confirmation_model, confirmation_metrics = self.train_confirmation_model(
                X_train, y_train, X_val, y_val
            )
            
            # Save both models
            warning_model_obj = MyModel(
                trained_model_object=warning_model,
                decision_threshold=warning_metrics["threshold"],
                model_type="warning"
            )
            
            confirmation_model_obj = MyModel(
                trained_model_object=confirmation_model,
                decision_threshold=confirmation_metrics["threshold"],
                model_type="confirmation"
            )
            
            # Save models
            save_object(
                self.model_trainer_config.trained_model_warning_path,
                warning_model_obj
            )
            
            save_object(
                self.model_trainer_config.trained_model_confirmation_path,
                confirmation_model_obj
            )
            
            # Save metrics for evaluation
            combined_metrics = {
                "warning_model": warning_metrics,
                "confirmation_model": confirmation_metrics
            }
            
            # Save metrics as JSON
            import json
            metrics_path = os.path.join(self.model_trainer_config.trained_model_dir, "training_metrics.json")

            
            # ✅ FIX: Convert numpy types to Python native types
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # Apply conversion
            serializable_metrics = convert_numpy_types(combined_metrics)
            
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
            
            logging.info("Dual models trained and saved successfully")
            
            return ModelTrainerArtifact(
                trained_warning_model_path=self.model_trainer_config.trained_model_warning_path,
                trained_confirmation_model_path=self.model_trainer_config.trained_model_confirmation_path,
                warning_model_auc=warning_metrics["roc_auc"],
                confirmation_model_auc=confirmation_metrics["roc_auc"],
                message="Dual sepsis prediction models trained successfully",
            )
            
        except Exception as e:
            raise MyException(e, sys)