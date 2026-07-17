# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest


class TestHealthAPI:
    def test_health_check(self):
        from app.main import create_app
        app = create_app()
        client = app.test_client()
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert 'status' in data


class TestPredictionAPI:
    def test_index_page(self):
        from app.main import create_app
        app = create_app()
        client = app.test_client()
        response = client.get('/')
        assert response.status_code == 200

    def test_api_predict(self):
        import json
        from app.main import create_app
        app = create_app()
        client = app.test_client()
        data = {
            'assignment_1': 80, 'assignment_2': 75, 'assignment_3': 85,
            'attendance': 90, 'study_hours': 12, 'past_gpa': 3.2
        }
        response = client.post('/api/predict', data=json.dumps(data), content_type='application/json')
        assert response.status_code == 200
        result = response.get_json()
        assert 'predictions' in result
