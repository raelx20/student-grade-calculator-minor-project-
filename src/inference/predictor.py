import os
import numpy as np
import joblib
import pandas as pd
from src.utils.constants import SCORE_MODEL, PASS_MODEL, SCALER_PATH, FEATURE_COLUMNS
from src.features.feature_engineering import add_features
from src.training.train_grade import map_to_grade
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    def __init__(self, score_model_path=None, pass_model_path=None, scaler_path=None):
        self.score_model_path = score_model_path or SCORE_MODEL
        self.pass_model_path = pass_model_path or PASS_MODEL
        self.scaler_path = scaler_path or SCALER_PATH
        self.score_model = joblib.load(self.score_model_path)
        self.pass_model = joblib.load(self.pass_model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.feature_cols = FEATURE_COLUMNS + ["avg_assignment_score", "assignment_trend", "study_attendance_interaction", "weighted_score", "performance_index"]
        logger.info("Predictor loaded successfully")

    def _prepare_features(self, data):
        df = pd.DataFrame([data])
        df = add_features(df)
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def predict_score(self, data):
        X = self._prepare_features(data)
        score = self.score_model.predict(X)[0]
        return round(float(np.clip(score, 0, 100)), 1)

    def predict_pass_fail(self, data):
        X = self._prepare_features(data)
        return int(self.pass_model.predict(X)[0])

    def predict_grade(self, data):
        score = self.predict_score(data)
        return map_to_grade(score)

    def predict_all(self, data):
        score = self.predict_score(data)
        pass_fail = self.predict_pass_fail(data)
        grade = map_to_grade(score)
        return {
            "predicted_score": score,
            "pass_fail": "Pass" if pass_fail else "Fail",
            "grade": grade,
        }