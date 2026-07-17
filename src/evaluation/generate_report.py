import os
import json
import pandas as pd
from src.utils.constants import BASE_DIR
from src.utils.logger import get_logger
from src.evaluation.visualization import (plot_correlation_heatmap, plot_feature_importance,
    plot_regression_results, plot_residuals, plot_confusion_matrix)

logger = get_logger(__name__)
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "reports", "metrics")


def generate_full_report(df, y_true_score, y_pred_score, y_true_pass, y_pred_pass, feature_names, importances_df):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    plot_correlation_heatmap(df, os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
    if not importances_df.empty:
        plot_feature_importance(importances_df, os.path.join(FIGURES_DIR, "feature_importance.png"))
    plot_regression_results(y_true_score, y_pred_score, os.path.join(FIGURES_DIR, "regression_plot.png"))
    plot_residuals(y_true_score, y_pred_score, os.path.join(FIGURES_DIR, "residual_plot.png"))
    plot_confusion_matrix(y_true_pass, y_pred_pass, os.path.join(FIGURES_DIR, "confusion_matrix.png"))
    logger.info("All report figures generated")