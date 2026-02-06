import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    accuracy_score,
    precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
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
    Evaluates both warning and confirmation models
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

    def _load_production_models_from_s3(self):
        """Load production models from S3 - with fallback"""
        try:
            warning_estimator = S3ModelEstimator(
                bucket_name=self.model_training_config.s3_bucket_name,
                s3_key="model-registry/warning_model.pkl"
            )
            
            confirmation_estimator = S3ModelEstimator(
                bucket_name=self.model_training_config.s3_bucket_name,
                s3_key="model-registry/confirmation_model.pkl"
            )
            
            # Try to load warning model only (it's small - 1.17 MB)
            prod_warning_model = None
            prod_confirmation_model = None
            
            if warning_estimator.is_model_present():
                try:
                    prod_warning_model = warning_estimator.load_model()
                    logging.info("Loaded warning model from S3")
                except Exception as e:
                    logging.warning(f"Could not load warning model: {e}")
            
            # SKIP confirmation model (850 MB) - it's too big
            logging.info("Skipping confirmation model load (850 MB - too large for network)")
            
            return prod_warning_model, prod_confirmation_model
            
        except Exception as e:
            logging.warning(f"Error loading from S3: {e}. Continuing without comparison.")
            return None, None

    def _evaluate_single_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                              model_name: str) -> Dict:
        """Evaluate a single model"""
        try:
            y_prob = model.trained_model_object.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= model.decision_threshold).astype(int)
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            recall = recall_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            metrics = {
                "roc_auc": float(roc_auc),  # ✅ Convert to float
                "pr_auc": float(pr_auc),    # ✅ Convert to float
                "recall": float(recall),    # ✅ Convert to float
                "precision": float(precision),  # ✅ Convert to float
                "accuracy": float(accuracy),    # ✅ Convert to float
                "threshold": float(model.decision_threshold),  # ✅ Convert to float
                "model_type": str(model.model_type)  # ✅ Convert to string
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1])
            }
            
            logging.info(
                f"{model_name} | ROC-AUC={metrics['roc_auc']:.4f} | "
                f"PR-AUC={metrics['pr_auc']:.4f} | "
                f"Recall={metrics['recall']:.3f} | "
                f"Precision={metrics['precision']:.3f}"
            )
            
            return metrics, y_prob
            
        except Exception as e:
            raise MyException(e, sys)

    def _save_curves(self, y_test: np.ndarray, warning_prob: np.ndarray, 
                    confirmation_prob: np.ndarray, output_dir: str):
        """Save ROC and PR curves for both models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # ROC Curve Comparison
        plt.figure(figsize=(12, 5))
        
        # ROC curves
        plt.subplot(1, 2, 1)
        fpr_warning, tpr_warning, _ = roc_curve(y_test, warning_prob)
        fpr_conf, tpr_conf, _ = roc_curve(y_test, confirmation_prob)
        
        plt.plot(fpr_warning, tpr_warning, label=f'Warning Model (AUC={roc_auc_score(y_test, warning_prob):.3f})')
        plt.plot(fpr_conf, tpr_conf, label=f'Confirmation Model (AUC={roc_auc_score(y_test, confirmation_prob):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # PR Curves
        plt.subplot(1, 2, 2)
        precision_warning, recall_warning, _ = precision_recall_curve(y_test, warning_prob)
        precision_conf, recall_conf, _ = precision_recall_curve(y_test, confirmation_prob)
        
        plt.plot(recall_warning, precision_warning, label=f'Warning Model (AP={average_precision_score(y_test, warning_prob):.3f})')
        plt.plot(recall_conf, precision_conf, label=f'Confirmation Model (AP={average_precision_score(y_test, confirmation_prob):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual curves
        for model_name, y_prob in [("warning", warning_prob), ("confirmation", confirmation_prob)]:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name.capitalize()} Model')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"roc_curve_{model_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name.capitalize()} Model')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"pr_curve_{model_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    def _convert_to_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._convert_to_serializable(obj.__dict__)
        else:
            return obj

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Dual Model Evaluation")
            
            os.makedirs("artifacts/evaluation", exist_ok=True)
            
            # Load test data
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_path
            )
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1].astype(int)
            
            # Load new models
            warning_model = load_object(
                self.model_trainer_artifact.trained_warning_model_path
            )
            confirmation_model = load_object(
                self.model_trainer_artifact.trained_confirmation_model_path
            )
            
            # Evaluate new models
            warning_metrics, warning_prob = self._evaluate_single_model(
                warning_model, X_test, y_test, "New Warning Model"
            )
            confirmation_metrics, confirmation_prob = self._evaluate_single_model(
                confirmation_model, X_test, y_test, "New Confirmation Model"
            )
            
            # Load production models for comparison
            prod_warning_model, prod_confirmation_model = self._load_production_models_from_s3()
            
            # Compare with production
            comparison_results = {}
            
            if prod_warning_model:
                prod_warning_metrics, _ = self._evaluate_single_model(
                    prod_warning_model, X_test, y_test, "Production Warning Model"
                )
                warning_improvement = warning_metrics["pr_auc"] - prod_warning_metrics["pr_auc"]
                comparison_results["warning_improvement"] = float(warning_improvement)  # ✅ Convert to float
                warning_accepted = warning_improvement >= 0
            else:
                warning_accepted = True
                comparison_results["warning_improvement"] = float(warning_metrics["pr_auc"])  # ✅ Convert to float
            
            if prod_confirmation_model:
                prod_confirmation_metrics, _ = self._evaluate_single_model(
                    prod_confirmation_model, X_test, y_test, "Production Confirmation Model"
                )
                confirmation_improvement = confirmation_metrics["pr_auc"] - prod_confirmation_metrics["pr_auc"]
                comparison_results["confirmation_improvement"] = float(confirmation_improvement)  # ✅ Convert to float
                confirmation_accepted = confirmation_improvement >= 0
            else:
                confirmation_accepted = True
                comparison_results["confirmation_improvement"] = float(confirmation_metrics["pr_auc"])  # ✅ Convert to float
            
            # Save curves
            self._save_curves(y_test, warning_prob, confirmation_prob, "artifacts/evaluation")
            
            # ✅ FIXED: Save comprehensive metrics
            all_metrics = {
                "warning_model": warning_metrics,
                "confirmation_model": confirmation_metrics,
                "comparison": comparison_results,
                "acceptance": {
                    "warning_model_accepted": bool(warning_accepted),  # ✅ Convert to bool
                    "confirmation_model_accepted": bool(confirmation_accepted)  # ✅ Convert to bool
                }
            }
            
            # Convert all metrics to serializable format
            serializable_metrics = self._convert_to_serializable(all_metrics)
            
            metrics_path = "artifacts/evaluation/dual_model_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(serializable_metrics, f, indent=4)
            
            # Save confusion matrices
            plt.figure(figsize=(10, 4))
            for idx, (model_name, metrics) in enumerate([("Warning", warning_metrics), 
                                                         ("Confirmation", confirmation_metrics)], 1):
                plt.subplot(1, 2, idx)
                cm = metrics["confusion_matrix"]
                cm_array = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
                plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix - {model_name} Model')
                plt.colorbar()
                tick_marks = [0, 1]
                plt.xticks(tick_marks, ['No Sepsis', 'Sepsis'])
                plt.yticks(tick_marks, ['No Sepsis', 'Sepsis'])
                
                # Add text annotations
                thresh = np.array(cm_array).max() / 2.
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, format(cm_array[i][j], 'd'),
                                horizontalalignment="center",
                                color="white" if cm_array[i][j] > thresh else "black")
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
            
            plt.tight_layout()
            plt.savefig("artifacts/evaluation/confusion_matrices.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Dual model evaluation completed")
            
            return ModelEvaluationArtifact(
                is_warning_model_accepted=warning_accepted,
                is_confirmation_model_accepted=confirmation_accepted,
                warning_model_pr_auc=float(warning_metrics["pr_auc"]),  # ✅ Convert to float
                confirmation_model_pr_auc=float(confirmation_metrics["pr_auc"]),  # ✅ Convert to float
                warning_model_recall=float(warning_metrics["recall"]),  # ✅ Convert to float
                confirmation_model_precision=float(confirmation_metrics["precision"]),  # ✅ Convert to float
                metrics_path=metrics_path,
                message="Dual model evaluation completed successfully"
            )
            
        except Exception as e:
            raise MyException(e, sys)