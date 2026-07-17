import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


def encode_grade(df: pd.DataFrame) -> tuple:
    grade_order = {'A+': 7, 'A': 6, 'B+': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
    df['grade_encoded'] = df['grade'].map(grade_order)
    return df, grade_order


def encode_categorical(df: pd.DataFrame, columns: list) -> tuple:
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info('Encoded %s: %d classes', col, len(le.classes_))
    return df, encoders
