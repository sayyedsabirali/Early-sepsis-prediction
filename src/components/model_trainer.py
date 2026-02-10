import os
import sys
import numpy as np
import xgboost as xgb
from typing import Tuple, Dict, List
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from src.constants import (
    XGBOOST_PARAMS,
    EXTRA_TREES_PARAMS,
    WARNING_TARGET_RECALL,
    CONFIRMATION_TARGET_PRECISION,
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.estimator import MyModel


class ModelTrainer:
    """
    Trains TWO models for sepsis prediction with threshold optimization:
    1. Warning Model (High Recall) - XGBoost with low threshold
    2. Confirmation Model (High Precision) - ExtraTrees with high threshold
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_metric: str = 'recall',
        target_value: float = None,
        secondary_metric: str = 'precision'
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold based on target metric
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            target_metric: 'recall' or 'precision'
            target_value: Target value for the metric
            secondary_metric: Metric to maximize when target is met
        
        Returns:
            Optimal threshold and metrics dictionary
        """
        thresholds = np.linspace(0.01, 0.99, 100)
        best_threshold = 0.5
        best_secondary_score = -1
        best_primary_score = -1
        target_achieved = False
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate all metrics
            recall_val = recall_score(y_true, y_pred, zero_division=0)
            precision_val = precision_score(y_true, y_pred, zero_division=0)
            f1_val = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'recall': recall_val,
                'precision': precision_val,
                'f1': f1_val
            })
            
            if target_metric == 'recall':
                primary = recall_val
                secondary = precision_val
            else:  # precision
                primary = precision_val
                secondary = recall_val
            
            # If target value is specified
            if target_value is not None:
                if primary >= target_value:
                    if not target_achieved or secondary > best_secondary_score:
                        target_achieved = True
                        best_threshold = threshold
                        best_secondary_score = secondary
                        best_primary_score = primary
                elif not target_achieved and primary > best_primary_score:
                    best_threshold = threshold
                    best_primary_score = primary
                    best_secondary_score = secondary
            else:
                # Maximize F1 score if no target
                if f1_val > best_secondary_score:
                    best_threshold = threshold
                    best_secondary_score = f1_val
        
        # Get metrics at best threshold
        y_pred_best = (y_prob >= best_threshold).astype(int)
        final_recall = recall_score(y_true, y_pred_best, zero_division=0)
        final_precision = precision_score(y_true, y_pred_best, zero_division=0)
        final_f1 = f1_score(y_true, y_pred_best, zero_division=0)
        
        return best_threshold, {
            'threshold': best_threshold,
            'recall': final_recall,
            'precision': final_precision,
            'f1': final_f1,
            'target_achieved': target_achieved,
            'all_results': results
        }

    def train_warning_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[object, Dict]:
        """Train XGBoost model for high recall with threshold optimization"""
        try:
            logging.info("\n" + "="*60)
            logging.info("TRAINING WARNING MODEL (XGBoost - High Recall)")
            logging.info("="*60)
            
            # Calculate class distribution
            neg = np.sum(y_train == 0)
            pos = np.sum(y_train == 1)
            imbalance_ratio = neg / max(pos, 1)
            
            logging.info(f"Training Data Class Distribution:")
            logging.info(f"  Negative (Non-sepsis): {neg:,} samples")
            logging.info(f"  Positive (Sepsis): {pos:,} samples")
            logging.info(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
            logging.info(f"  Positive Rate: {(pos/(neg+pos))*100:.2f}%")
            
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = imbalance_ratio
            logging.info(f"\nUsing scale_pos_weight: {scale_pos_weight:.2f}")
            
            # Enhanced parameters for imbalanced data
            params = {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": -1,
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
                "eval_metric": ["logloss", "auc", "error"],
                "use_label_encoder": False,
                "learning_rate": 0.05,  # Lower learning rate for stability
                "max_depth": 8,  # Slightly deeper for complex patterns
                "min_child_weight": 5,  # Higher to prevent overfitting
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,  # L1 regularization
                "reg_lambda": 1.0,  # L2 regularization
                "gamma": 0.1  # Minimum loss reduction
            }
            
            # Train model with early stopping
            logging.info("\nTraining XGBoost model...")
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100,  # Show progress every 100 trees
            )
            
            # Get probabilities on VALIDATION set for threshold tuning
            y_val_prob = model.predict_proba(X_val)[:, 1]
            
            # Find optimal threshold for high recall
            logging.info("\nOptimizing threshold for Warning Model...")
            logging.info(f"Target Recall: {WARNING_TARGET_RECALL}")
            
            best_threshold, threshold_metrics = self.find_optimal_threshold(
                y_val, y_val_prob,
                target_metric='recall',
                target_value=WARNING_TARGET_RECALL,
                secondary_metric='precision'
            )
            
            # Also check default threshold (0.5) and very low thresholds
            default_threshold = 0.5
            low_thresholds = [0.1, 0.2, 0.3, 0.4]
            
            logging.info("\nThreshold Analysis (Validation Set):")
            logging.info("-" * 80)
            
            for thresh in [default_threshold, best_threshold] + low_thresholds:
                if thresh in low_thresholds and abs(thresh - best_threshold) < 0.01:
                    continue
                    
                y_val_pred = (y_val_prob >= thresh).astype(int)
                recall = recall_score(y_val, y_val_pred, zero_division=0)
                precision = precision_score(y_val, y_val_pred, zero_division=0)
                f1 = f1_score(y_val, y_val_pred, zero_division=0)
                
                mark = " * " if thresh == best_threshold else "   "
                logging.info(f"{mark}Threshold {thresh:.3f}: "
                           f"Recall={recall:.4f}, "
                           f"Precision={precision:.4f}, "
                           f"F1={f1:.4f}")
            
            logging.info("-" * 80)
            
            # Evaluate on TEST set with optimized threshold
            logging.info(f"\nEvaluating on TEST SET with threshold={best_threshold:.3f}")
            y_test_prob = model.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_prob >= best_threshold).astype(int)
            
            # Also get default threshold predictions for comparison
            y_test_pred_default = (y_test_prob >= default_threshold).astype(int)
            
            # Calculate metrics at optimized threshold
            test_roc_auc = roc_auc_score(y_test, y_test_prob)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensitivity
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            
            # Metrics at default threshold
            test_recall_default = recall_score(y_test, y_test_pred_default, zero_division=0)
            test_precision_default = precision_score(y_test, y_test_pred_default, zero_division=0)
            
            # Validation metrics
            y_val_pred_opt = (y_val_prob >= best_threshold).astype(int)
            val_recall = recall_score(y_val, y_val_pred_opt, zero_division=0)
            val_precision = precision_score(y_val, y_val_pred_opt, zero_division=0)
            val_f1 = f1_score(y_val, y_val_pred_opt, zero_division=0)
            
            # Log detailed performance
            logging.info("\n" + "="*60)
            logging.info("WARNING MODEL PERFORMANCE SUMMARY")
            logging.info("="*60)
            logging.info(f"Optimized Threshold: {best_threshold:.3f}")
            logging.info(f"Target Recall Achieved: {test_recall}")
            logging.info("\nTest Set Metrics:")
            logging.info(f"Recall (Sensitivity):{test_recall:.4f}")
            logging.info(f"  Precision (PPV):         {test_precision:.4f}")
            logging.info(f"  F1-Score:                {test_f1:.4f}")
            logging.info(f"  ROC-AUC:                 {test_roc_auc:.4f}")
            logging.info(f"  Accuracy:                {test_accuracy:.4f}")
            logging.info(f"  False Positive Rate:     {fpr:.4f}")
            
            logging.info("\nConfusion Matrix (Test Set):")
            logging.info(f"  True Positives (TP):  {tp}")
            logging.info(f"  False Positives (FP): {fp}")
            logging.info(f"  True Negatives (TN):  {tn}")
            logging.info(f"  False Negatives (FN): {fn}")
            
            logging.info("\nComparison with Default Threshold (0.5):")
            logging.info(f"  Recall:    {test_recall_default:.4f} -> {test_recall:.4f} "
                        f"(Delta: {test_recall - test_recall_default:+.4f})")
            logging.info(f"  Precision: {test_precision_default:.4f} -> {test_precision:.4f} "
                        f"(Delta: {test_precision - test_precision_default:+.4f})")
            
            return model, {
                "test_roc_auc": test_roc_auc,
                "test_recall": test_recall,
                "test_precision": test_precision,
                "test_f1": test_f1,
                "test_accuracy": test_accuracy,
                "val_recall": val_recall,
                "val_precision": val_precision,
                "val_f1": val_f1,
                "threshold": best_threshold,
                "model_type": "xgboost_warning",
                "scale_pos_weight": scale_pos_weight,
                "imbalance_ratio": imbalance_ratio,
                "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "fpr": fpr,
                "target_achieved": test_recall >= WARNING_TARGET_RECALL
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
        """Train optimized ExtraTrees model for high precision with threshold optimization"""
        
        try:
            logging.info("\n" + "="*60)
            logging.info("TRAINING CONFIRMATION MODEL (ExtraTrees - High Precision)")
            logging.info("="*60)
            
            # Calculate class distribution
            neg = np.sum(y_train == 0)
            pos = np.sum(y_train == 1)
            imbalance_ratio = neg / max(pos, 1)
            
            logging.info(f"Training Data Class Distribution:")
            logging.info(f"  Negative (Non-sepsis): {neg:,} samples")
            logging.info(f"  Positive (Sepsis): {pos:,} samples")
            logging.info(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
            
            # Calculate class weights for imbalanced data
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
            
            logging.info(f"\nUsing class weights: {class_weight_dict}")
            
            # Use params from constants with class weights
            params = EXTRA_TREES_PARAMS.copy()
            params['class_weight'] = class_weight_dict
            params['n_estimators'] = 200  # Increase for better performance
            params['min_samples_leaf'] = 5  # Prevent overfitting
            params['max_features'] = 'sqrt'  # Better for high-dimensional data
            
            logging.info("\nTraining ExtraTrees model...")
            model = ExtraTreesClassifier(**params)
            model.fit(X_train, y_train)
            
            # Feature importance
            feature_importance = model.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-5:]  # Top 5 features
            logging.info(f"\nTop 5 Feature Importances:")
            for idx in top_features_idx[::-1]:
                logging.info(f"  Feature {idx}: {feature_importance[idx]:.4f}")
            
            # Get probabilities on VALIDATION set for threshold tuning
            y_val_prob = model.predict_proba(X_val)[:, 1]
            
            # Find optimal threshold for high precision
            logging.info("\nOptimizing threshold for Confirmation Model...")
            logging.info(f"Target Precision: {CONFIRMATION_TARGET_PRECISION}")
            
            best_threshold, threshold_metrics = self.find_optimal_threshold(
                y_val, y_val_prob,
                target_metric='precision',
                target_value=CONFIRMATION_TARGET_PRECISION,
                secondary_metric='recall'
            )
            
            # Check high thresholds for precision
            high_thresholds = [0.6, 0.7, 0.8, 0.9]
            default_threshold = 0.5
            
            logging.info("\nThreshold Analysis (Validation Set):")
            logging.info("-" * 80)
            
            for thresh in [default_threshold, best_threshold] + high_thresholds:
                if thresh in high_thresholds and abs(thresh - best_threshold) < 0.01:
                    continue
                    
                y_val_pred = (y_val_prob >= thresh).astype(int)
                precision = precision_score(y_val, y_val_pred, zero_division=0)
                recall = recall_score(y_val, y_val_pred, zero_division=0)
                f1 = f1_score(y_val, y_val_pred, zero_division=0)
                
                mark = " * " if thresh == best_threshold else "   "
                logging.info(f"{mark}Threshold {thresh:.3f}: "
                           f"Precision={precision:.4f}, "
                           f"Recall={recall:.4f}, "
                           f"F1={f1:.4f}")
            
            logging.info("-" * 80)
            
            # Evaluate on TEST set with optimized threshold
            logging.info(f"\nEvaluating on TEST SET with threshold={best_threshold:.3f}")
            y_test_prob = model.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_prob >= best_threshold).astype(int)
            
            # Calculate metrics
            test_roc_auc = roc_auc_score(y_test, y_test_prob)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate Positive Predictive Value (same as precision)
            ppv = test_precision
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            
            # Validation metrics
            y_val_pred_opt = (y_val_prob >= best_threshold).astype(int)
            val_precision = precision_score(y_val, y_val_pred_opt, zero_division=0)
            val_recall = recall_score(y_val, y_val_pred_opt, zero_division=0)
            val_f1 = f1_score(y_val, y_val_pred_opt, zero_division=0)
            
            # Log detailed performance - Unicode symbols replace with ASCII
            logging.info("\n" + "="*60)
            logging.info("CONFIRMATION MODEL PERFORMANCE SUMMARY")
            logging.info("="*60)
            logging.info(f"Optimized Threshold: {best_threshold:.3f}")
            logging.info(f"Target Precision Achieved: {test_precision >= CONFIRMATION_TARGET_PRECISION}")
            logging.info("\nTest Set Metrics:")
            logging.info(f"  Precision (PPV):         {test_precision:.4f}  (Target: >= {CONFIRMATION_TARGET_PRECISION})")
            logging.info(f"  Recall (Sensitivity):    {test_recall:.4f}")
            logging.info(f"  F1-Score:                {test_f1:.4f}")
            logging.info(f"  ROC-AUC:                 {test_roc_auc:.4f}")
            logging.info(f"  Accuracy:                {test_accuracy:.4f}")
            logging.info(f"  NPV:                     {npv:.4f}")
            
            logging.info("\nConfusion Matrix (Test Set):")
            logging.info(f"  True Positives (TP):  {tp}")
            logging.info(f"  False Positives (FP): {fp}")
            logging.info(f"  True Negatives (TN):  {tn}")
            logging.info(f"  False Negatives (FN): {fn}")
            
            # Calculate alerts per case
            total_predictions = len(y_test_pred)
            positive_predictions = np.sum(y_test_pred == 1)
            alert_rate = (positive_predictions / total_predictions) * 100
            
            logging.info(f"\nAlert Statistics:")
            logging.info(f"  Total Cases: {total_predictions}")
            logging.info(f"  Positive Alerts: {positive_predictions}")
            logging.info(f"  Alert Rate: {alert_rate:.2f}%")
            
            return model, {
                "test_roc_auc": test_roc_auc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_accuracy": test_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "threshold": best_threshold,
                "model_type": "extratrees_confirmation",
                "class_weights": class_weight_dict,
                "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "alert_rate": alert_rate,
                "npv": npv,
                "target_achieved": test_precision >= CONFIRMATION_TARGET_PRECISION
            }
            
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Main training pipeline for dual models with threshold optimization"""
        try:
            logging.info("\n" + "="*80)
            logging.info("STARTING DUAL MODEL TRAINING PIPELINE")
            logging.info("="*80)
            logging.info(f"Warning Model Target Recall: >= {WARNING_TARGET_RECALL}")
            logging.info(f"Confirmation Model Target Precision: >= {CONFIRMATION_TARGET_PRECISION}")
            
            # Load ALL data
            logging.info("\nLoading transformed data...")
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
            
            # Log data statistics
            logging.info("\nData Statistics:")
            logging.info(f"Training data:   {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
            logging.info(f"Validation data: {X_val.shape[0]:,} samples")
            logging.info(f"Test data:       {X_test.shape[0]:,} samples")
            
            total_pos = np.sum(y_train) + np.sum(y_val) + np.sum(y_test)
            total_neg = len(y_train) + len(y_val) + len(y_test) - total_pos
            logging.info(f"\nOverall Class Distribution:")
            logging.info(f"  Positive (Sepsis): {total_pos:,} samples")
            logging.info(f"  Negative (Non-sepsis): {total_neg:,} samples")
            logging.info(f"  Imbalance Ratio: {total_neg/total_pos:.1f}:1")
            
            # Train Warning Model (XGBoost) - High Recall
            warning_model, warning_metrics = self.train_warning_model(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Train Confirmation Model (ExtraTrees) - High Precision
            confirmation_model, confirmation_metrics = self.train_confirmation_model(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Create model objects with optimized thresholds
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
            
            # Save models locally
            logging.info("\nSaving trained models...")
            save_object(
                self.model_trainer_config.trained_model_warning_path,
                warning_model_obj
            )
            
            save_object(
                self.model_trainer_config.trained_model_confirmation_path,
                confirmation_model_obj
            )
            
            # Create and return artifact with updated parameters
            artifact = ModelTrainerArtifact(
                trained_warning_model_path=self.model_trainer_config.trained_model_warning_path,
                trained_confirmation_model_path=self.model_trainer_config.trained_model_confirmation_path,
                warning_model_auc=warning_metrics["test_roc_auc"],
                confirmation_model_auc=confirmation_metrics["test_roc_auc"],
                warning_model_recall=warning_metrics["test_recall"],  # ✅ Added
                confirmation_model_precision=confirmation_metrics["test_precision"],  # ✅ Added
                warning_threshold=warning_metrics["threshold"],  # ✅ Added
                confirmation_threshold=confirmation_metrics["threshold"],  # ✅ Added
                message=(
                    f"Dual models trained successfully. "
                    f"Warning Model (Recall={warning_metrics['test_recall']:.4f}, "
                    f"Threshold={warning_metrics['threshold']:.3f}), "
                    f"Confirmation Model (Precision={confirmation_metrics['test_precision']:.4f}, "
                    f"Threshold={confirmation_metrics['threshold']:.3f})"
                )
            )
            
            # Final summary - Unicode symbols replace with ASCII
            logging.info("\n" + "="*80)
            logging.info("DUAL MODEL TRAINING COMPLETED SUCCESSFULLY!")
            logging.info("="*80)
            
            logging.info("\nFINAL MODEL PERFORMANCE SUMMARY:")
            logging.info("-" * 80)
            
            logging.info("WARNING MODEL (XGBoost - High Recall):")
            logging.info(f"  Purpose: Early detection, minimize false negatives")
            logging.info(f"  Threshold: {warning_metrics['threshold']:.3f}")
            # Replace ✓✗ with PASS/FAIL
            target_status = "PASS" if warning_metrics['test_recall'] >= WARNING_TARGET_RECALL else "FAIL"
            logging.info(f"  Recall:    {warning_metrics['test_recall']:.4f} "
                        f"(Target: >= {WARNING_TARGET_RECALL}) {target_status}")
            logging.info(f"  Precision: {warning_metrics['test_precision']:.4f}")
            logging.info(f"  F1-Score:  {warning_metrics['test_f1']:.4f}")
            logging.info(f"  ROC-AUC:   {warning_metrics['test_roc_auc']:.4f}")
            logging.info(f"  FPR:       {warning_metrics['fpr']:.4f}")
            
            logging.info("\nCONFIRMATION MODEL (ExtraTrees - High Precision):")
            logging.info(f"  Purpose: Confirmation, minimize false positives")
            logging.info(f"  Threshold: {confirmation_metrics['threshold']:.3f}")
            # Replace ✓✗ with PASS/FAIL
            target_status = "PASS" if confirmation_metrics['test_precision'] >= CONFIRMATION_TARGET_PRECISION else "FAIL"
            logging.info(f"  Precision: {confirmation_metrics['test_precision']:.4f} "
                        f"(Target: >= {CONFIRMATION_TARGET_PRECISION}) {target_status}")
            logging.info(f"  Recall:    {confirmation_metrics['test_recall']:.4f}")
            logging.info(f"  F1-Score:  {confirmation_metrics['test_f1']:.4f}")
            logging.info(f"  ROC-AUC:   {confirmation_metrics['test_roc_auc']:.4f}")
            logging.info(f"  Alert Rate: {confirmation_metrics['alert_rate']:.2f}%")
            
            logging.info("\nMODEL DEPLOYMENT READY:")
            logging.info(f"  1. Warning Model saved: {self.model_trainer_config.trained_model_warning_path}")
            logging.info(f"  2. Confirmation Model saved: {self.model_trainer_config.trained_model_confirmation_path}")
            
            # Usage recommendation
            logging.info("\nUSAGE RECOMMENDATION:")
            logging.info("  1. Use Warning Model for initial screening")
            logging.info("  2. Use Confirmation Model for final verification")
            logging.info("  3. Combined system provides balanced sensitivity and specificity")
            
            logging.info("=" * 80)
            
            return artifact
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise MyException(e, sys)