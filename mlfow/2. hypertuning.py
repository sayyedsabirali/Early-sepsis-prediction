import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# ========== DATA LOADING ==========
print("Loading Sepsis Data for Model Tuning...")
print("="*60)

def load_data(npy_path):
    arr = np.load(npy_path)
    X = arr[:, :-1]
    y = arr[:, -1].astype(int)
    return X, y

X_train, y_train = load_data('artifacts/data_transformation/train.npy')
X_test, y_test = load_data('artifacts/data_transformation/test.npy')
X_val, y_val = load_data('artifacts/data_transformation/val.npy')

print(f"Data loaded: Train={X_train.shape}, Test={X_test.shape}")

# ========== SUBSAMPLE FOR TUNING (FASTER) ==========
from sklearn.model_selection import train_test_split

# Take 500k samples for tuning (faster)
tune_sample_size = min(500000, len(X_train))
X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train, 
    train_size=tune_sample_size, 
    stratify=y_train,
    random_state=42
)
print(f"Using {len(X_tune):,} samples for hyperparameter tuning")

# ========== CONFIRMATION MODELS (HIGH PRECISION) ==========
print(f"\n{'='*70}")
print("TUNING CONFIRMATION MODELS (High Precision Required)")
print(f"{'='*70}")

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier

confirmation_models = {
    "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
}

# Hyperparameter grids for CONFIRMATION (optimize for precision)
param_grids = {
    "ExtraTrees": {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    
    "RandomForest": {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, 10, 50, 100]
    }
}

# MLflow setup
mlflow.set_experiment("Sepsis_Confirmation_Models_Tuned")

confirmation_results = []

for model_name, model in confirmation_models.items():
    print(f"\nTuning {model_name} for CONFIRMATION (High Precision)...")
    
    with mlflow.start_run(run_name=f"Tuned_{model_name}"):
        try:
            start_time = time.time()
            
            # Randomized Search with 5-fold CV
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grids[model_name],
                n_iter=10,  # Reduced for speed
                scoring='precision',  # OPTIMIZE FOR PRECISION!
                cv=cv,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
            
            print(f"  Starting hyperparameter tuning...")
            random_search.fit(X_tune, y_tune)
            tuning_time = time.time() - start_time
            
            # Get best model
            best_model = random_search.best_estimator_
            
            # Train on full data
            print(f"  Training on full dataset...")
            best_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
            
            # Metrics
            metrics = {
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision': precision_score(y_test, y_pred, zero_division=0),
                'test_recall': recall_score(y_test, y_pred, zero_division=0),
                'test_f1': f1_score(y_test, y_pred, zero_division=0),
                'test_specificity': None  # Will calculate below
            }
            
            # Calculate specificity
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics['test_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            if y_proba is not None:
                metrics['test_roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log best parameters
            for param, value in random_search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)
            
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("tuning_time", tuning_time)
            mlflow.log_param("n_samples_tuned", len(X_tune))
            
            # Save model
            mlflow.sklearn.log_model(best_model, "tuned_model")
            
            confirmation_results.append({
                'Model': model_name,
                'Best_Precision': metrics['test_precision'],
                'Recall': metrics['test_recall'],
                'F1_Score': metrics['test_f1'],
                'Specificity': metrics['test_specificity'],
                'ROC_AUC': metrics.get('test_roc_auc', np.nan),
                'Tuning_Time': tuning_time,
                'Best_Params': random_search.best_params_
            })
            
            print(f"  {model_name} tuning completed in {tuning_time:.2f}s")
            print(f"     Best Precision: {metrics['test_precision']:.4f}")
            print(f"     Recall: {metrics['test_recall']:.4f}")
            print(f"     Specificity: {metrics['test_specificity']:.4f}")
            
        except Exception as e:
            print(f"  Error tuning {model_name}: {str(e)}")
            continue

# ========== WARNING MODELS (HIGH RECALL) ==========
print(f"\n{'='*70}")
print("TUNING WARNING MODELS (High Recall Required)")
print(f"{'='*70}")

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warning_models = {
    "XGBoost_Warning": XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
    "LightGBM_Warning": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    "CatBoost_Warning": CatBoostClassifier(random_state=42, verbose=0)
}

# Hyperparameter grids for WARNING (optimize for recall)
warning_param_grids = {
    "XGBoost_Warning": {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'scale_pos_weight': [50, 100, 200],  # Higher for better recall
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    },
    
    "LightGBM_Warning": {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'min_child_samples': [20, 50, 100],
        'is_unbalance': [True],
        'boosting_type': ['gbdt', 'dart']
    },
    
    "CatBoost_Warning": {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 10],
        'auto_class_weights': ['Balanced', 'SqrtBalanced'],
        'bootstrap_type': ['Bayesian', 'Bernoulli']
    }
}

mlflow.set_experiment("Sepsis_Warning_Models_Tuned")

warning_results = []

for model_name, model in warning_models.items():
    print(f"\nTuning {model_name} for WARNING (High Recall)...")
    
    with mlflow.start_run(run_name=f"Tuned_{model_name}"):
        try:
            start_time = time.time()
            
            # Randomized Search with recall scoring
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=warning_param_grids[model_name],
                n_iter=10,
                scoring='recall',  # OPTIMIZE FOR RECALL!
                cv=cv,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
            
            print(f"  Starting hyperparameter tuning...")
            random_search.fit(X_tune, y_tune)
            tuning_time = time.time() - start_time
            
            # Get best model
            best_model = random_search.best_estimator_
            
            # Train on full data
            print(f"  Training on full dataset...")
            best_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
            
            # Metrics
            metrics = {
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision': precision_score(y_test, y_pred, zero_division=0),
                'test_recall': recall_score(y_test, y_pred, zero_division=0),
                'test_f1': f1_score(y_test, y_pred, zero_division=0),
            }
            
            if y_proba is not None:
                metrics['test_roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log best parameters
            for param, value in random_search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)
            
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("tuning_time", tuning_time)
            mlflow.log_param("scoring", "recall")
            
            # Save model
            mlflow.sklearn.log_model(best_model, "tuned_warning_model")
            
            warning_results.append({
                'Model': model_name,
                'Precision': metrics['test_precision'],
                'Best_Recall': metrics['test_recall'],
                'F1_Score': metrics['test_f1'],
                'ROC_AUC': metrics.get('test_roc_auc', np.nan),
                'Tuning_Time': tuning_time,
                'Best_Params': random_search.best_params_
            })
            
            print(f"  {model_name} tuning completed in {tuning_time:.2f}s")
            print(f"     Best Recall: {metrics['test_recall']:.4f}")
            print(f"     Precision: {metrics['test_precision']:.4f}")
            print(f"     F1-Score: {metrics['test_f1']:.4f}")
            
        except Exception as e:
            print(f"  Error tuning {model_name}: {str(e)}")
            continue

# ========== FINAL SELECTION ==========
print(f"\n{'='*80}")
print("FINAL MODEL SELECTION")
print(f"{'='*80}")

# Select best confirmation model (highest precision)
if confirmation_results:
    confirmation_df = pd.DataFrame(confirmation_results)
    best_confirmation = confirmation_df.loc[confirmation_df['Best_Precision'].idxmax()]
    
    print(f"\nBEST CONFIRMATION MODEL:")
    print(f"   Model: {best_confirmation['Model']}")
    print(f"   Precision: {best_confirmation['Best_Precision']:.4f}")
    print(f"   Specificity: {best_confirmation['Specificity']:.4f}")
    print(f"   F1-Score: {best_confirmation['F1_Score']:.4f}")

# Select best warning model (highest recall)
if warning_results:
    warning_df = pd.DataFrame(warning_results)
    best_warning = warning_df.loc[warning_df['Best_Recall'].idxmax()]
    
    print(f"\nBEST WARNING MODEL:")
    print(f"   Model: {best_warning['Model']}")
    print(f"   Recall: {best_warning['Best_Recall']:.4f}")
    print(f"   Precision: {best_warning['Precision']:.4f}")
    print(f"   F1-Score: {best_warning['F1_Score']:.4f}")

# Save results
if confirmation_results:
    pd.DataFrame(confirmation_results).to_csv('confirmation_models_tuned.csv', index=False)
    print(f"\nConfirmation models saved to: confirmation_models_tuned.csv")

if warning_results:
    pd.DataFrame(warning_results).to_csv('warning_models_tuned.csv', index=False)
    print(f"Warning models saved to: warning_models_tuned.csv")

print(f"\nMLflow Instructions:")
print("1. Confirmation Models: mlflow ui --port 5000 (experiment: Sepsis_Confirmation_Models_Tuned)")
print("2. Warning Models: mlflow ui --port 5001 (experiment: Sepsis_Warning_Models_Tuned)")
print("\nDeployment Strategy:")
print("   - Use WARNING model first (high recall - catch all possible sepsis)")
print("   - Use CONFIRMATION model on warnings (high precision - confirm true sepsis)")
print(f"\nProcess completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")  