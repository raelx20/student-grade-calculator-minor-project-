import pandas as pd
from src.inference.predictor import Predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def batch_predict(csv_path, predictor=None):
    if predictor is None:
        predictor = Predictor()
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        data = row.to_dict()
        pred = predictor.predict_all(data)
        results.append(pred)
    result_df = pd.DataFrame(results)
    output = pd.concat([df, result_df], axis=1)
    return output


def save_batch_results(df, output_path):
    df.to_csv(output_path, index=False)