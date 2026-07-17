import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_regression_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    metrics = {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}
    logger.info("Regression: R2=%.4f MAE=%.4f RMSE=%.4f", r2, mae, rmse)
    return metrics


def print_regression_report(metrics):
    print("Regression Metrics:")
    for k, v in metrics.items():
        print("  " + k.upper() + ": " + str(v))