import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.constants import FEATURE_COLUMNS, TARGET_PASS, RANDOM_SEED, TEST_SIZE, PASS_MODEL, SCALER_PATH
from src.utils.logger import get_logger
from src.features.feature_engineering import add_features
from src.evaluation.classification_metrics import compute_classification_metrics

logger = get_logger(__name__)


def train_pass_model():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'processed_students.csv'))
    df = add_features(df)
    feature_cols = FEATURE_COLUMNS + ['avg_assignment_score', 'assignment_trend', 'study_attendance_interaction', 'weighted_score', 'performance_index']
    X = df[feature_cols].values
    y = df[TARGET_PASS].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    logger.info('Pass Model Metrics: Acc=%.4f, F1=%.4f, AUC=%.4f', metrics['accuracy'], metrics['f1'], metrics['auc_roc'])

    os.makedirs(os.path.dirname(PASS_MODEL), exist_ok=True)
    joblib.dump(model, PASS_MODEL)
    joblib.dump(scaler, SCALER_PATH)
    logger.info('Saved pass model and scaler')
    return model, scaler, metrics


if __name__ == '__main__':
    train_pass_model()
