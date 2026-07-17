import os
import joblib
from src.utils.constants import SCORE_MODEL, PASS_MODEL, SCALER_PATH, ENCODER_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_all_models(score_model, pass_model, scaler, encoder=None):
    os.makedirs(os.path.dirname(SCORE_MODEL), exist_ok=True)
    joblib.dump(score_model, SCORE_MODEL)
    joblib.dump(pass_model, PASS_MODEL)
    joblib.dump(scaler, SCALER_PATH)
    if encoder:
        joblib.dump(encoder, ENCODER_PATH)
    logger.info('All models saved')


def load_all_models():
    models = {
        'score_model': joblib.load(SCORE_MODEL),
        'pass_model': joblib.load(PASS_MODEL),
        'scaler': joblib.load(SCALER_PATH),
    }
    if os.path.exists(ENCODER_PATH):
        models['encoder'] = joblib.load(ENCODER_PATH)
    logger.info('All models loaded')
    return models
