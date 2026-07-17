# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
import numpy as np


class TestScoreTraining:
    def test_train_score_model(self):
        from src.training.train_score import train_score_model
        model, scaler, metrics = train_score_model()
        assert metrics['r2'] > 0
        assert metrics['mae'] > 0


class TestPassTraining:
    def test_train_pass_model(self):
        from src.training.train_pass import train_pass_model
        model, scaler, metrics = train_pass_model()
        assert metrics['accuracy'] > 0
        assert metrics['f1'] > 0


class TestGradeMapping:
    def test_map_to_grade(self):
        from src.training.train_grade import map_to_grade
        assert map_to_grade(95) == 'A+'
        assert map_to_grade(85) == 'A'
        assert map_to_grade(75) == 'B+'
        assert map_to_grade(65) == 'B'
        assert map_to_grade(55) == 'C'
        assert map_to_grade(45) == 'D'
        assert map_to_grade(30) == 'F'
