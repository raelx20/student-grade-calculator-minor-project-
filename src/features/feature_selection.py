import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        if importance.ndim > 1:
            importance = importance.mean(axis=0)
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return pd.DataFrame()
    df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    return df


def remove_collinear(df: pd.DataFrame, threshold: float = 0.9) -> list:
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info('Dropping %d collinear features: %s', len(to_drop), to_drop)
    return to_drop
