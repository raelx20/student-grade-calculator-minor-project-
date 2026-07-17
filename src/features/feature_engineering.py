import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    assignment_cols = ['assignment_1', 'assignment_2', 'assignment_3']
    df['avg_assignment_score'] = df[assignment_cols].mean(axis=1).round(2)
    df['assignment_trend'] = (df['assignment_3'] - df['assignment_1']).round(2)
    df['study_attendance_interaction'] = (df['study_hours'] * df['attendance'] / 100).round(2)
    df['weighted_score'] = (
        0.25 * df['assignment_1'] +
        0.25 * df['assignment_2'] +
        0.30 * df['assignment_3'] +
        0.10 * df['attendance'] +
        0.10 * df['study_hours'] * 2
    ).round(2)
    df['performance_index'] = (
        0.5 * df['past_gpa'] / 10.0 * 100 +
        0.5 * df['avg_assignment_score']
    ).round(2)
    logger.info('Added 5 engineered features')
    return df
