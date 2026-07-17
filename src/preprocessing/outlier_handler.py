import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            col_mask = (df[col] < lower) | (df[col] > upper)
            mask = mask | col_mask
            logger.info('%s: %d outliers (IQR bounds: %.2f - %.2f)', col, col_mask.sum(), lower, upper)
    return mask


def detect_outliers_zscore(df: pd.DataFrame, columns: list, threshold: float = 3.0) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_mask = z > threshold
            mask = mask | col_mask
    return mask


def handle_outliers(df: pd.DataFrame, columns: list, method: str = 'cap') -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            if method == 'cap':
                before_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                df[col] = df[col].clip(lower=lower, upper=upper)
                logger.info('Capped %d outliers in %s', before_outliers, col)
            elif method == 'remove':
                df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df
