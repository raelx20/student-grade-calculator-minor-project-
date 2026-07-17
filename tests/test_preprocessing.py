# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
import pandas as pd
import numpy as np


class TestLoader:
    def test_load_data(self):
        from src.preprocessing.loader import load_data
        df = load_data('data/raw/students.csv')
        assert len(df) > 0
        assert 'final_score' in df.columns


class TestValidator:
    def test_validate_schema(self):
        from src.preprocessing.validator import validate_schema
        df = pd.DataFrame({
            'assignment_1': [80], 'assignment_2': [70], 'assignment_3': [90],
            'attendance': [85], 'study_hours': [10], 'past_gpa': [3.0],
            'final_score': [82], 'pass_fail': [1], 'grade': ['A']
        })
        result = validate_schema(df)
        assert result['valid'] == True


class TestCleaner:
    def test_remove_duplicates(self):
        from src.preprocessing.cleaner import remove_duplicates
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4]})
        result = remove_duplicates(df)
        assert len(result) == 2


class TestMissingValues:
    def test_handle_missing(self):
        from src.preprocessing.missing_values import handle_missing
        df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, 6]})
        result = handle_missing(df)
        assert result['a'].isnull().sum() == 0


class TestOutlierHandler:
    def test_handle_outliers(self):
        from src.preprocessing.outlier_handler import handle_outliers
        df = pd.DataFrame({'a': [1, 2, 3, 100, 5]})
        result = handle_outliers(df, ['a'])
        assert result['a'].max() <= 100


class TestFeatureEngineering:
    def test_add_features(self):
        from src.features.feature_engineering import add_features
        df = pd.DataFrame({
            'assignment_1': [80], 'assignment_2': [70], 'assignment_3': [90],
            'attendance': [85], 'study_hours': [10], 'past_gpa': [3.0]
        })
        result = add_features(df)
        assert 'avg_assignment_score' in result.columns
        assert 'weighted_score' in result.columns
