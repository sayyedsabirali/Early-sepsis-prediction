import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC


def get_models(scale_pos_weight):
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss"
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingClassifier(),

        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000
        ),

        "SVM": SVC(
            probability=True,
            class_weight="balanced"
        )
    }
