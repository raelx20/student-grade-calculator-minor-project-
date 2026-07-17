from src.inference.predictor import Predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    _predictor = None

    @classmethod
    def _get_predictor(cls):
        if cls._predictor is None:
            try:
                cls._predictor = Predictor()
                logger.info("Predictor initialized")
            except Exception as e:
                logger.error("Failed to load predictor: %s", str(e))
                raise
        return cls._predictor

    def predict(self, data):
        predictor = self._get_predictor()
        return predictor.predict_all(data)