import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import tempfile
import os
import time
from tqdm import tqdm

# ========== DATA LOADING ==========
print("Loading Sepsis Data...")
print("="*60)

def load_sepsis_data(npy_path):
    """Load numpy array and split into features and labels"""
    arr = np.load(npy_path)
    X = arr[:, :-1]  # First 12 columns are features
    y = arr[:, -1].astype(int)  # Last column is target (0/1)
    return X, y

# Load all data
start_time = time.time()
X_train, y_train = load_sepsis_data('artifacts/data_transformation/train.npy')
X_test, y_test = load_sepsis_data('artifacts/data_transformation/test.npy')  
X_val, y_val = load_sepsis_data('artifacts/data_transformation/val.npy')
load_time = time.time() - start_time

print(f"Data loaded in {load_time:.2f} seconds")
print(f"X_train: {X_train.shape} ({X_train.shape[0]:,} samples, {X_train.shape[1]} features)")
print(f"X_test: {X_test.shape} ({X_test.shape[0]:,} samples)")
print(f"X_val: {X_val.shape} ({X_val.shape[0]:,} samples)")

print(f"\nClass Distribution:")
for cls in [0, 1]:
    train_count = (y_train == cls).sum()
    test_count = (y_test == cls).sum()
    val_count = (y_val == cls).sum()
    print(f"  Class {cls}: Train={train_count:,} ({train_count/len(y_train)*100:.2f}%) | "
          f"Test={test_count:,} | Val={val_count:,}")

print(f"\nImbalanced dataset! Using appropriate techniques...")

# ========== MODELS FOR LARGE DATASET ==========
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

# Models optimized for large datasets
models = {
    # Fast linear models for large data
    "LogisticRegression_SGD": SGDClassifier(
        loss='log_loss', 
        max_iter=1000, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    ),
    
    "LinearSVC": LinearSVC(
        max_iter=1000, 
        random_state=42, 
        class_weight='balanced',
        dual=False
    ),
    
    # Tree-based (efficient for large data)
    "RandomForest_100": RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced_subsample'
    ),
    
    "ExtraTrees_100": ExtraTreesClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced_subsample'
    ),
    
    # Gradient Boosting (memory efficient)
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=100, 
        random_state=42,
        early_stopping=True
    ),
    
    "LightGBM": LGBMClassifier(
        n_estimators=100, 
        random_state=42, 
        verbose=-1, 
        n_jobs=-1,
        is_unbalance=True,
        boosting_type='gbdt'
    ),
    
    "XGBoost": XGBClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        tree_method='hist',  # Memory efficient
        eval_metric='logloss'
    ),
    
    "CatBoost": CatBoostClassifier(
        iterations=100, 
        random_state=42, 
        verbose=0,
        auto_class_weights='Balanced',
        task_type='CPU'
    ),
    
    # Simple models
    "NaiveBayes": GaussianNB(),
    
    "DecisionTree": DecisionTreeClassifier(
        random_state=42, 
        class_weight='balanced'
    ),
    
    "AdaBoost": AdaBoostClassifier(
        n_estimators=50, 
        random_state=42
    ),
}

print(f"\nTraining {len(models)} models on FULL dataset...")
print("This will take time due to 5.4M+ samples...")

# ========== HELPER FUNCTIONS ==========
def save_plot_to_temp(fig, plot_name):
    """Save plot to temp file"""
    temp_dir = tempfile.mkdtemp()
    plot_path = os.path.join(temp_dir, f"{plot_name}.png")
    fig.savefig(plot_path, bbox_inches='tight', dpi=100, facecolor='white')
    plt.close(fig)
    return plot_path

# ========== MLFLOW SETUP ==========
mlflow.set_experiment("Sepsis_Full_Training")

# ========== TRAIN & LOG ==========
results_summary = []

for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Training {model_name} on {X_train.shape[0]:,} samples")
    print(f"{'='*70}")
    
    run_start_time = time.time()
    
    with mlflow.start_run(run_name=model_name):
        try:
            # Train model with progress indicator
            print(f"  Training started...")
            train_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - train_start
            print(f"  Training completed in {train_time:.2f} seconds")
            
            # Predictions
            print(f"  Making predictions...")
            y_pred_test = model.predict(X_test)
            y_pred_val = model.predict(X_val)
            
            # Probabilities (if available)
            if hasattr(model, 'predict_proba'):
                y_proba_test = model.predict_proba(X_test)
            else:
                y_proba_test = None
            
            # ========== CALCULATE METRICS ==========
            metrics = {}
            
            # Basic metrics
            metrics['test_accuracy'] = accuracy_score(y_test, y_pred_test)
            metrics['test_precision'] = precision_score(y_test, y_pred_test, zero_division=0)
            metrics['test_recall'] = recall_score(y_test, y_pred_test, zero_division=0)
            metrics['test_f1'] = f1_score(y_test, y_pred_test, zero_division=0)
            
            metrics['val_accuracy'] = accuracy_score(y_val, y_pred_val)
            metrics['val_precision'] = precision_score(y_val, y_pred_val, zero_division=0)
            metrics['val_recall'] = recall_score(y_val, y_pred_val, zero_division=0)
            metrics['val_f1'] = f1_score(y_val, y_pred_val, zero_division=0)
            
            # AUC metrics if probabilities available
            if y_proba_test is not None:
                metrics['test_roc_auc'] = roc_auc_score(y_test, y_proba_test[:, 1])
                metrics['test_avg_precision'] = average_precision_score(y_test, y_proba_test[:, 1])
                
                # Validation probabilities
                if hasattr(model, 'predict_proba'):
                    y_proba_val = model.predict_proba(X_val)
                    metrics['val_roc_auc'] = roc_auc_score(y_val, y_proba_val[:, 1])
                    metrics['val_avg_precision'] = average_precision_score(y_val, y_proba_val[:, 1])
            
            # Specificity
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
            metrics['test_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # ========== LOG TO MLFLOW ==========
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("train_features", X_train.shape[1])
            mlflow.log_param("training_time_seconds", train_time)
            
            # Log model
            mlflow.sklearn.log_model(model, "trained_model")
            
            # ========== CREATE & SAVE PLOTS ==========
            
            # 1. Confusion Matrix
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred_test)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                       xticklabels=['No Sepsis', 'Sepsis'],
                       yticklabels=['No Sepsis', 'Sepsis'],
                       annot_kws={"size": 16})
            ax1.set_title(f'Confusion Matrix - {model_name}\nTest Set ({X_test.shape[0]:,} samples)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax1.set_ylabel('True Label', fontsize=12)
            ax1.set_xlabel('Predicted Label', fontsize=12)
            ax1.tick_params(axis='both', which='major', labelsize=10)
            
            cm_plot_path = save_plot_to_temp(fig1, f"confusion_matrix_{model_name}")
            mlflow.log_artifact(cm_plot_path, "plots")
            
            # 2. ROC Curve (if probabilities available)
            if y_proba_test is not None and 'test_roc_auc' in metrics:
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                fpr, tpr, _ = roc_curve(y_test, y_proba_test[:, 1])
                roc_auc = metrics['test_roc_auc']
                
                ax2.plot(fpr, tpr, color='darkorange', lw=3, 
                        label=f'ROC Curve (AUC = {roc_auc:.4f})')
                ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
                ax2.fill_between(fpr, tpr, alpha=0.2, color='orange')
                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel('False Positive Rate', fontsize=12)
                ax2.set_ylabel('True Positive Rate', fontsize=12)
                ax2.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold', pad=20)
                ax2.legend(loc="lower right", fontsize=12)
                ax2.grid(True, alpha=0.3)
                
                roc_plot_path = save_plot_to_temp(fig2, f"roc_curve_{model_name}")
                mlflow.log_artifact(roc_plot_path, "plots")
            
            # 3. Feature Importance (if available)
            if hasattr(model, 'feature_importances_'):
                try:
                    importances = model.feature_importances_
                    if len(importances) > 0:
                        # Get top 15 features
                        n_top = min(15, len(importances))
                        indices = np.argsort(importances)[-n_top:]
                        
                        fig3, ax3 = plt.subplots(figsize=(12, 10))
                        y_pos = np.arange(len(indices))
                        ax3.barh(y_pos, importances[indices])
                        ax3.set_yticks(y_pos)
                        ax3.set_yticklabels([f'Feature {i}' for i in indices], fontsize=10)
                        ax3.set_xlabel('Importance Score', fontsize=12)
                        ax3.set_title(f'Top {n_top} Feature Importances - {model_name}', 
                                     fontsize=14, fontweight='bold', pad=20)
                        ax3.grid(True, axis='x', alpha=0.3)
                        
                        # Add value labels
                        for i, v in enumerate(importances[indices]):
                            ax3.text(v, i, f' {v:.4f}', va='center', fontsize=9)
                        
                        fi_plot_path = save_plot_to_temp(fig3, f"feature_importance_{model_name}")
                        mlflow.log_artifact(fi_plot_path, "plots")
                except Exception as e:
                    print(f"  Could not create feature importance plot: {e}")
            
            # ========== SAVE REPORTS ==========
            
            # Classification Report
            report = classification_report(y_test, y_pred_test, 
                                          target_names=['No Sepsis', 'Sepsis'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_path = f"{model_name}_classification_report.csv"
            report_df.to_csv(report_path, index=True)
            mlflow.log_artifact(report_path)
            
            # Detailed metrics file
            metrics_file = f"{model_name}_detailed_metrics.txt"
            with open(metrics_file, 'w') as f:
                f.write(f"{'='*60}\n")
                f.write(f"MODEL: {model_name}\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Training Configuration:\n")
                f.write(f"  Samples: {X_train.shape[0]:,}\n")
                f.write(f"  Features: {X_train.shape[1]}\n")
                f.write(f"  Training Time: {train_time:.2f} seconds\n\n")
                f.write(f"Test Set Performance:\n")
                f.write(f"  Accuracy: {metrics['test_accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['test_precision']:.4f}\n")
                f.write(f"  Recall: {metrics['test_recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['test_f1']:.4f}\n")
                f.write(f"  Specificity: {metrics['test_specificity']:.4f}\n")
                if 'test_roc_auc' in metrics:
                    f.write(f"  ROC-AUC: {metrics['test_roc_auc']:.4f}\n")
                    f.write(f"  Average Precision: {metrics['test_avg_precision']:.4f}\n")
                f.write(f"\nValidation Set Performance:\n")
                f.write(f"  Accuracy: {metrics['val_accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['val_precision']:.4f}\n")
                f.write(f"  Recall: {metrics['val_recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['val_f1']:.4f}\n")
            
            mlflow.log_artifact(metrics_file)
            
            # ========== UPDATE RESULTS ==========
            results_summary.append({
                'Model': model_name,
                'Train_Samples': X_train.shape[0],
                'Train_Time_sec': train_time,
                'Test_Accuracy': metrics['test_accuracy'],
                'Test_Precision': metrics['test_precision'],
                'Test_Recall': metrics['test_recall'],
                'Test_F1': metrics['test_f1'],
                'Test_Specificity': metrics['test_specificity'],
                'Test_ROC_AUC': metrics.get('test_roc_auc', np.nan),
                'Test_Avg_Precision': metrics.get('test_avg_precision', np.nan),
                'Val_Accuracy': metrics['val_accuracy'],
                'Val_F1': metrics['val_f1'],
            })
            
            run_time = time.time() - run_start_time
            print(f"  {model_name} completed in {run_time:.2f} seconds")
            print(f"     Test Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"     Test F1-Score: {metrics['test_f1']:.4f}")
            if 'test_roc_auc' in metrics:
                print(f"     Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
            print(f"     Test Precision: {metrics['test_precision']:.4f}")
            print(f"     Test Recall: {metrics['test_recall']:.4f}")
            
        except Exception as e:
            run_time = time.time() - run_start_time
            print(f"  {model_name} failed after {run_time:.2f} seconds: {str(e)}")
            continue

# ========== FINAL SUMMARY ==========
print(f"\n{'='*80}")
print("ALL MODELS TRAINING COMPLETED!")
print(f"{'='*80}")

if results_summary:
    results_df = pd.DataFrame(results_summary)
    
    # Sort by ROC-AUC if available, otherwise by F1
    if 'Test_ROC_AUC' in results_df.columns and not results_df['Test_ROC_AUC'].isna().all():
        results_df = results_df.sort_values('Test_ROC_AUC', ascending=False)
        sort_by = "ROC-AUC"
    else:
        results_df = results_df.sort_values('Test_F1', ascending=False)
        sort_by = "F1-Score"
    
    print(f"\nModel Performance Summary (Sorted by {sort_by}):")
    print("-"*80)
    
    # Display only key columns
    display_cols = ['Model', 'Train_Time_sec', 'Test_Accuracy', 'Test_Precision', 
                   'Test_Recall', 'Test_F1', 'Test_ROC_AUC']
    display_cols = [col for col in display_cols if col in results_df.columns]
    
    print(results_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"))
    
    # Best model
    best_model = results_df.iloc[0]
    print(f"\nBEST MODEL: {best_model['Model']}")
    print(f"   Test F1-Score: {best_model['Test_F1']:.4f}")
    print(f"   Test Accuracy: {best_model['Test_Accuracy']:.4f}")
    if 'Test_ROC_AUC' in best_model and pd.notnull(best_model['Test_ROC_AUC']):
        print(f"   Test ROC-AUC: {best_model['Test_ROC_AUC']:.4f}")
    print(f"   Training Time: {best_model['Train_Time_sec']:.2f} seconds")
    
    # Save full results
    results_df.to_csv('sepsis_models_full_training_results.csv', index=False)
    print(f"\nFull results saved to: sepsis_models_full_training_results.csv")
    
    # Calculate and display training statistics
    total_training_time = results_df['Train_Time_sec'].sum()
    avg_training_time = results_df['Train_Time_sec'].mean()
    print(f"\nTraining Statistics:")
    print(f"   Total training time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
    print(f"   Average per model: {avg_training_time:.2f} seconds")
    print(f"   Total samples trained: {X_train.shape[0]:,}")
    print(f"   Successful models: {len(results_df)}/{len(models)}")

print(f"\nMLflow Instructions:")
print("1. Open NEW terminal/command prompt")
print("2. Run: mlflow ui --port 5000")
print("3. Open browser: http://localhost:5000")
print("4. Select 'Sepsis_Full_Training' experiment")
print(f"\nEach model run contains:")
print("   • All performance metrics")
print("   • Confusion matrix plot")
print("   • ROC curve plot (if available)")
print("   • Feature importance plot (if available)")
print("   • Classification report")
print("   • Detailed metrics file")
print("   • Trained model artifact")

print(f"\nProcess completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")  