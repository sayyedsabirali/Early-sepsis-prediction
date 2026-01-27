import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    idx = np.argmax(f1)
    return thresholds[idx]


def evaluate(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    threshold = find_best_threshold(y, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
        "best_threshold": float(threshold)
    }, y_prob
