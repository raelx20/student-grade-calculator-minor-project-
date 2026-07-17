# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest


class TestPredictor:
    def test_predict_all(self):
        from src.inference.predictor import Predictor
        p = Predictor()
        data = {
            'assignment_1': 80, 'assignment_2': 75, 'assignment_3': 85,
            'attendance': 90, 'study_hours': 12, 'past_gpa': 3.2
        }
        result = p.predict_all(data)
        assert 'predicted_score' in result
        assert 'pass_fail' in result
        assert 'grade' in result
        assert 0 <= result['predicted_score'] <= 100


class TestRecommendationEngine:
    def test_generate_recommendations(self):
        from src.inference.recommender import RecommendationEngine
        engine = RecommendationEngine()
        data = {'attendance': 50, 'study_hours': 5, 'assignment_1': 40, 'assignment_2': 45, 'assignment_3': 50, 'past_gpa': 2.0}
        predictions = {'predicted_score': 45}
        recs = engine.generate_recommendations(data, predictions)
        assert len(recs) > 0
        assert all('area' in r for r in recs)
        assert all('message' in r for r in recs)
