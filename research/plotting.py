import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


def save_curves(y, y_prob, name, folder):
    # PR Curve
    precision, recall, _ = precision_recall_curve(y, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"{name} PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.savefig(f"{folder}/{name}_pr_curve.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"{name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid()
    plt.savefig(f"{folder}/{name}_roc_curve.png")
    plt.close()
