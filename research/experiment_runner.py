import os
import joblib
import pandas as pd
import numpy as np

from research.models import get_models
from research.evaluator import evaluate
from research.plotting import save_curves
from src.utils.preprocessing_utils import PreprocessingUtils


TRAIN_PATH = "artifacts/data_ingestion/train.csv"
TEST_PATH  = "artifacts/data_ingestion/test.csv"
TARGET = "label_12h"

RESULT_DIR = "research/results"
os.makedirs(RESULT_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X = PreprocessingUtils.apply_preprocessing_transformations(X)
    return X.values, y.values


def main():
    print("Loading data...")
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test   = load_data(TEST_PATH)

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / max(pos, 1)

    models = get_models(scale_pos_weight)
    results = []

    print("\n=== MODEL EXPERIMENTS STARTED ===")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        metrics, y_prob = evaluate(model, X_test, y_test)
        print(f"{name} → {metrics}")

        # Save model
        joblib.dump(model, f"{RESULT_DIR}/{name}.pkl")

        # Save curves
        save_curves(y_test, y_prob, name, RESULT_DIR)

        metrics["model"] = name
        results.append(metrics)

    pd.DataFrame(results).to_csv(f"{RESULT_DIR}/model_comparison.csv", index=False)

    print("\nAll models, metrics, and plots saved in research/results/")


if __name__ == "__main__":
    main()
