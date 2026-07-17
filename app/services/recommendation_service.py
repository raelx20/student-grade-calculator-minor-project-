from src.inference.recommender import RecommendationEngine


class RecommendationService:
    _engine = None

    @classmethod
    def _get_engine(cls):
        if cls._engine is None:
            cls._engine = RecommendationEngine()
        return cls._engine

    def get_recommendations(self, data, predictions):
        engine = self._get_engine()
        return engine.generate_recommendations(data, predictions)