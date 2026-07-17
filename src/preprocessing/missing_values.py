import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_missing(df):
    missing = df.isnull().sum()
    return {col: count for col, count in missing.items() if count > 0}


def handle_missing(df, strategy='median'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if strategy == 'median':
                fill_val = df[col].median()
            elif strategy == 'mean':
                fill_val = df[col].mean()
            else:
                fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            logger.info('Filled %s missing with %s (%.2f)', col, strategy, fill_val)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            logger.info('Filled %s missing with mode: %s', col, fill_val)
    return df


def drop_rows_with_critical_missing(df, columns):
    before = len(df)
    df = df.dropna(subset=columns)
    dropped = before - len(df)
    if dropped > 0:
        logger.info('Dropped %d rows with missing critical values', dropped)
    return df
