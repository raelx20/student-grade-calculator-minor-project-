import pandas as pd
from src.utils.constants import FEATURE_COLUMNS, TARGET_SCORE, TARGET_PASS, TARGET_GRADE, CGPA_MIN, CGPA_MAX
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_schema(df):
    required = FEATURE_COLUMNS + [TARGET_SCORE, TARGET_PASS, TARGET_GRADE]
    missing = [c for c in required if c not in df.columns]
    issues = []
    if missing:
        issues.append('Missing columns: ' + str(missing))
    return {'valid': len(issues) == 0, 'issues': issues}


def validate_ranges(df):
    issues = []
    for col in ['assignment_1', 'assignment_2', 'assignment_3', 'attendance']:
        if col in df.columns:
            invalid = df[(df[col] < 0) | (df[col] > 100)]
            if len(invalid) > 0:
                issues.append(col + ': ' + str(len(invalid)) + ' rows out of [0,100]')
    if 'past_gpa' in df.columns:
        invalid = df[(df['past_gpa'] < CGPA_MIN) | (df['past_gpa'] > CGPA_MAX)]
        if len(invalid) > 0:
            issues.append('past_gpa: ' + str(len(invalid)) + ' rows out of [' + str(CGPA_MIN) + ',' + str(CGPA_MAX) + ']')
    return {'valid': len(issues) == 0, 'issues': issues}
