import os
import mlflow
import dagshub
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix,accuracy_score, f1_score, precision_score, recall_score, classification_report,roc_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
warnings.filterwarnings('ignore')

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "https://dagshub.com/sayyedsabirali/Early-sepsis-prediction.mlflow"
dagshub.init(repo_owner='sayyedsabirali', repo_name='Early-sepsis-prediction', mlflow=True)

# Paths
TRAIN_PATH = "artifacts/data_ingestion/train.csv"
VAL_PATH = "artifacts/data_ingestion/val.csv"
TEST_PATH = "artifacts/data_ingestion/test.csv"
TARGET_COLUMN = "label_12h"
RESULT_DIR = "research/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Import your preprocessing utilities
from src.utils.preprocessing_utils import PreprocessingUtils


def load_and_preprocess_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess data"""
    df = pd.read_csv(path)
    y = df[TARGET_COLUMN].values
    X = df.drop(columns=[TARGET_COLUMN])
    X = PreprocessingUtils.apply_preprocessing_transformations(X)
    return X.values, y


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal threshold using F1 score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores[:-1])  # Exclude last element
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


class ModelEvaluator:
    """Comprehensive model evaluation with MLflow tracking"""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate all evaluation metrics"""
        return {
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'specificity': ModelEvaluator.calculate_specificity(y_true, y_pred),
            'n_positives': int(sum(y_true)),
            'n_negatives': int(len(y_true) - sum(y_true)),
            'pos_rate': float(sum(y_true) / len(y_true))
        }
    
    @staticmethod
    def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    @staticmethod
    def evaluate_model(model, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str, params: Dict = None) -> Dict:
        """Evaluate model and return comprehensive results"""
        
        # Predict probabilities
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]
        
        # Find best threshold
        threshold = find_best_threshold(y_train, y_prob_train)
        
        # Make predictions
        y_pred_train = (y_prob_train >= threshold).astype(int)
        y_pred_test = (y_prob_test >= threshold).astype(int)
        
        # Calculate metrics
        train_metrics = ModelEvaluator.calculate_all_metrics(y_train, y_prob_train, y_pred_train)
        test_metrics = ModelEvaluator.calculate_all_metrics(y_test, y_prob_test, y_pred_test)
        
        # Cross-validation scores
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, 
                                   cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return {
            'model_name': model_name,
            'model': model,
            'threshold': threshold,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_prob_test': y_prob_test,
            'y_pred_test': y_pred_test,
            'params': params
        }


def get_all_models(scale_pos_weight: float, random_state: int = 42) -> Dict:
    """Define multiple models for experimentation"""
    
    models = {}
    
    # 1. XGBoost Variations
    models["XGBoost_Default"] = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1
    )
    
    models["XGBoost_Deep"] = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1
    )
    
    # 2. LightGBM Variations
    models["LightGBM_Default"] = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    
    models["LightGBM_Fast"] = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    
    # 3. CatBoost
    models["CatBoost"] = cb.CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_state=random_state,
        auto_class_weights='Balanced'
    )
    
    # 4. Random Forest Variations
    models["RandomForest_Default"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    models["RandomForest_Conservative"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=random_state,
        n_jobs=-1
    )
    
    # 5. Extra Trees
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    # 6. Gradient Boosting
    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=random_state
    )
    
    # 7. Logistic Regression Variations
    models["LogisticRegression_L2"] = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=2000,
        random_state=random_state,
        n_jobs=-1
    )
    
    models["LogisticRegression_L1"] = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        penalty='l1',
        solver='liblinear',
        max_iter=2000,
        random_state=random_state
    )
    
    # 8. SVM (for comparison)
    models["SVM_RBF"] = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=random_state,
        max_iter=1000
    )
    
    # 9. Balanced Random Forest
    from imblearn.ensemble import BalancedRandomForestClassifier
    models["BalancedRandomForest"] = BalancedRandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )
    
    # 10. Easy Ensemble
    from imblearn.ensemble import EasyEnsembleClassifier
    models["EasyEnsemble"] = EasyEnsembleClassifier(
        n_estimators=10,
        random_state=random_state,
        n_jobs=-1
    )
    
    return models


def plot_and_save_curves(y_true: np.ndarray, y_prob: np.ndarray, 
                        model_name: str, result_dir: str):
    """Create and save evaluation curves"""
    
    # Create subdirectory for this model
    model_dir = os.path.join(result_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(model_dir, f'{model_name}_roc_curve.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(model_dir, f'{model_name}_pr_curve.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Probability Distribution
    plt.figure(figsize=(12, 8))
    for label in [0, 1]:
        mask = y_true == label
        plt.hist(y_prob[mask], bins=50, alpha=0.5, 
                label=f'Class {label}', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(f'{model_name} - Probability Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(model_dir, f'{model_name}_prob_dist.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Calibration Curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'{model_name} - Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(model_dir, f'{model_name}_calibration_curve.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def log_to_mlflow(results: Dict, experiment_name: str = "Sepsis_Prediction"):
    """Log experiment results to MLflow"""
    
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=results['model_name']):
        # Log parameters
        if results['params']:
            mlflow.log_params(results['params'])
        
        # Log training metrics
        for key, value in results['train_metrics'].items():
            mlflow.log_metric(f"train_{key}", value)
        
        # Log test metrics
        for key, value in results['test_metrics'].items():
            mlflow.log_metric(f"test_{key}", value)
        
        # Log cross-validation results
        mlflow.log_metric("cv_roc_auc_mean", results['cv_mean'])
        mlflow.log_metric("cv_roc_auc_std", results['cv_std'])
        mlflow.log_metric("best_threshold", results['threshold'])
        
        # Log confusion matrix
        y_true = results.get('y_true', None)
        y_pred = results['y_pred_test']
        if y_true is not None:
            cm = confusion_matrix(y_true, y_pred)
            cm_fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {results["model_name"]}')
            mlflow.log_figure(cm_fig, "confusion_matrix.png")
            plt.close()
        
        # Log model
        mlflow.sklearn.log_model(results['model'], results['model_name'])
        
        # Log artifacts
        if os.path.exists(RESULT_DIR):
            mlflow.log_artifacts(RESULT_DIR, artifact_path="results")
        
        # Set tags
        mlflow.set_tags({
            "target": TARGET_COLUMN,
            "task": "binary_classification",
            "domain": "healthcare",
            "problem": "sepsis_prediction"
        })


def create_comparison_report(all_results: List[Dict], result_dir: str):
    """Create comprehensive comparison report"""
    
    # Extract metrics for comparison
    comparison_data = []
    
    for result in all_results:
        row = {
            'Model': result['model_name'],
            'ROC_AUC_Test': result['test_metrics']['roc_auc'],
            'PR_AUC_Test': result['test_metrics']['pr_auc'],
            'F1_Score_Test': result['test_metrics']['f1'],
            'Precision_Test': result['test_metrics']['precision'],
            'Recall_Test': result['test_metrics']['recall'],
            'Specificity_Test': result['test_metrics']['specificity'],
            'Accuracy_Test': result['test_metrics']['accuracy'],
            'ROC_AUC_Train': result['train_metrics']['roc_auc'],
            'CV_ROC_AUC_Mean': result['cv_mean'],
            'CV_ROC_AUC_Std': result['cv_std'],
            'Best_Threshold': result['threshold'],
            'Training_Positives': result['train_metrics']['n_positives'],
            'Test_Positives': result['test_metrics']['n_positives']
        }
        comparison_data.append(row)
    
    # Create DataFrame and save
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by ROC AUC
    comparison_df = comparison_df.sort_values('ROC_AUC_Test', ascending=False)
    
    # Save to CSV
    comparison_df.to_csv(os.path.join(result_dir, 'model_comparison.csv'), index=False)
    
    # Create visual comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ROC AUC Comparison
    axes[0, 0].barh(range(len(comparison_df)), comparison_df['ROC_AUC_Test'])
    axes[0, 0].set_yticks(range(len(comparison_df)))
    axes[0, 0].set_yticklabels(comparison_df['Model'])
    axes[0, 0].set_xlabel('ROC AUC')
    axes[0, 0].set_title('ROC AUC Comparison (Test)')
    axes[0, 0].invert_yaxis()
    
    # PR AUC Comparison
    axes[0, 1].barh(range(len(comparison_df)), comparison_df['PR_AUC_Test'])
    axes[0, 1].set_yticks(range(len(comparison_df)))
    axes[0, 1].set_yticklabels(comparison_df['Model'])
    axes[0, 1].set_xlabel('PR AUC')
    axes[0, 1].set_title('PR AUC Comparison (Test)')
    axes[0, 1].invert_yaxis()
    
    # F1 Score Comparison
    axes[1, 0].barh(range(len(comparison_df)), comparison_df['F1_Score_Test'])
    axes[1, 0].set_yticks(range(len(comparison_df)))
    axes[1, 0].set_yticklabels(comparison_df['Model'])
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_title('F1 Score Comparison (Test)')
    axes[1, 0].invert_yaxis()
    
    # Precision-Recall Comparison
    axes[1, 1].scatter(comparison_df['Recall_Test'], comparison_df['Precision_Test'])
    for i, row in comparison_df.iterrows():
        axes[1, 1].annotate(row['Model'], 
                          (row['Recall_Test'], row['Precision_Test']),
                          fontsize=8, alpha=0.7)
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Precision-Recall Trade-off (Test)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'model_comparison_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_df


def main():
    """Main training pipeline"""
    
    print("=" * 80)
    print("SEPSIS PREDICTION MODEL TRAINING")
    print("=" * 80)
    
    # Import models here to avoid circular imports
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                 ExtraTreesClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # Load data
    print("\n1. Loading and preprocessing data...")
    X_train, y_train = load_and_preprocess_data(TRAIN_PATH)
    X_val, y_val = load_and_preprocess_data(VAL_PATH) if os.path.exists(VAL_PATH) else (None, None)
    X_test, y_test = load_and_preprocess_data(TEST_PATH)
    
    print(f"   Train shape: {X_train.shape}, Positives: {sum(y_train)}/{len(y_train)}")
    print(f"   Test shape: {X_test.shape}, Positives: {sum(y_test)}/{len(y_test)}")
    
    # Calculate class weight
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / max(pos, 1)
    print(f"   Class balance - Positive: {pos}, Negative: {neg}")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    
    # Get all models
    print("\n2. Initializing models...")
    models = get_all_models(scale_pos_weight)
    print(f"   Total models: {len(models)}")
    
    # Train and evaluate all models
    print("\n3. Training and evaluating models...")
    all_results = []
    
    for model_name, model in models.items():
        try:
            print(f"\n   {'='*60}")
            print(f"   MODEL: {model_name}")
            print(f"   {'='*60}")
            
            # Train model
            print("   Training...")
            model.fit(X_train, y_train)
            
            # Evaluate
            print("   Evaluating...")
            results = ModelEvaluator.evaluate_model(
                model, X_train, y_train, X_test, y_test, model_name
            )
            results['y_true'] = y_test  # Store true labels for confusion matrix
            
            # Save plots
            print("   Creating plots...")
            plot_and_save_curves(y_test, results['y_prob_test'], model_name, RESULT_DIR)
            
            # Save model
            model_path = os.path.join(RESULT_DIR, model_name, f"{model_name}.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            
            # Log to MLflow
            print("   Logging to MLflow...")
            log_to_mlflow(results)
            
            all_results.append(results)
            
            # Print summary
            print(f"\n   Results for {model_name}:")
            print(f"   ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
            print(f"   PR AUC: {results['test_metrics']['pr_auc']:.4f}")
            print(f"   F1 Score: {results['test_metrics']['f1']:.4f}")
            print(f"   Best Threshold: {results['threshold']:.4f}")
            
        except Exception as e:
            print(f"   ERROR with {model_name}: {str(e)}")
            continue
    
    # Create comparison report
    print("\n4. Creating comparison report...")
    comparison_df = create_comparison_report(all_results, RESULT_DIR)
    
    # Save best model
    best_model_idx = np.argmax([r['test_metrics']['roc_auc'] for r in all_results])
    best_model_result = all_results[best_model_idx]
    best_model_name = best_model_result['model_name']
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest Model: {best_model_name}")
    print(f"ROC AUC: {best_model_result['test_metrics']['roc_auc']:.4f}")
    print(f"PR AUC: {best_model_result['test_metrics']['pr_auc']:.4f}")
    print(f"\nResults saved in: {RESULT_DIR}")
    print(f"Models logged to MLflow: {MLFLOW_TRACKING_URI}")
    
    # Display top 5 models
    print("\nTop 5 Models by ROC AUC:")
    print(comparison_df[['Model', 'ROC_AUC_Test', 'PR_AUC_Test', 'F1_Score_Test']].head(5).to_string(index=False))


if __name__ == "__main__":
    main()