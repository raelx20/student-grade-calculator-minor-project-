import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["auc_roc"] = round(roc_auc_score(y_true, y_prob), 4)
    logger.info("Classification: Acc=%.4f F1=%.4f", metrics["accuracy"], metrics["f1"])
    return metrics


def print_classification_report(metrics):
    print("Classification Metrics:")
    for k, v in metrics.items():
        print("  " + k.upper() + ": " + str(v))