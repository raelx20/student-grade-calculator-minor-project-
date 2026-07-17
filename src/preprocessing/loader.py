import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info('Loaded data: %d rows, %d columns', df.shape[0], df.shape[1])
    logger.info('Columns: %s', list(df.columns))
    return df
