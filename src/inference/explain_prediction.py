import numpy as np
import pandas as pd
from src.features.feature_engineering import add_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def explain_prediction(data, predictor):
    df = pd.DataFrame([data])
    df = add_features(df)
    feature_cols = predictor.feature_cols
    X = df[feature_cols].values
    X_scaled = predictor.scaler.transform(X)
    if hasattr(predictor.score_model, "coef_"):
        coef = predictor.score_model.coef_
        contributions = X_scaled[0] * coef
        explanations = []
        for i, feat in enumerate(feature_cols):
            explanations.append({"feature": feat, "value": float(X[0][i]), "contribution": round(float(contributions[i]), 2)})
        explanations.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return {"prediction": predictor.predict_all(data), "feature_contributions": explanations}
    return {"prediction": predictor.predict_all(data), "feature_contributions": []}