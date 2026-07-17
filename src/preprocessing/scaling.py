import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)


def scale_features(df: pd.DataFrame, columns: list, method: str = 'standard') -> tuple:
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    existing_cols = [c for c in columns if c in df.columns]
    df[existing_cols] = scaler.fit_transform(df[existing_cols])
    logger.info('Scaled %d features with %s scaler', len(existing_cols), method)
    return df, scaler
