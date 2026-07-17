import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.constants import FEATURE_COLUMNS, TARGET_SCORE, RANDOM_SEED, TEST_SIZE, SCORE_MODEL, SCALER_PATH
from src.utils.logger import get_logger
from src.features.feature_engineering import add_features
from src.evaluation.regression_metrics import compute_regression_metrics

logger = get_logger(__name__)


def train_score_model():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'processed_students.csv'))
    df = add_features(df)
    feature_cols = FEATURE_COLUMNS + ['avg_assignment_score', 'assignment_trend', 'study_attendance_interaction', 'weighted_score', 'performance_index']
    X = df[feature_cols].values
    y = df[TARGET_SCORE].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = compute_regression_metrics(y_test, y_pred)
    logger.info('Score Model Metrics: R2=%.4f, MAE=%.4f, RMSE=%.4f', metrics['r2'], metrics['mae'], metrics['rmse'])

    os.makedirs(os.path.dirname(SCORE_MODEL), exist_ok=True)
    joblib.dump(model, SCORE_MODEL)
    joblib.dump(scaler, SCALER_PATH)
    logger.info('Saved score model and scaler')
    return model, scaler, metrics


if __name__ == '__main__':
    train_score_model()
