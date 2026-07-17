import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        logger.info('Removed %d duplicate rows', removed)
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ['assignment_1', 'assignment_2', 'assignment_3', 'attendance', 'study_hours', 'past_gpa', 'final_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'pass_fail' in df.columns:
        df['pass_fail'] = df['pass_fail'].astype(int)
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df
