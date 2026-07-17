import time
import joblib
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_model(model, path: str):
    joblib.dump(model, path)
    logger.info('Model saved to %s', path)


def load_model(path: str):
    model = joblib.load(path)
    logger.info('Model loaded from %s', path)
    return model


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info('Loaded CSV: %s (%d rows, %d cols)', path, df.shape[0], df.shape[1])
    return df


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    logger.info('CSV saved to %s', path)


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info('%s completed in %.2fs', func.__name__, elapsed)
        return result
    return wrapper
