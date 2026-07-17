from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_regression_models():
    return {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
    }


def get_classification_models():
    return {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
    }


def select_best_model(models: dict, X, y, task: str = 'regression', cv: int = 5):
    results = {}
    scoring = 'r2' if task == 'regression' else 'accuracy'
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results[name] = {'mean': scores.mean(), 'std': scores.std()}
        logger.info('%s CV %s: %.4f (+/- %.4f)', name, scoring, scores.mean(), scores.std())
    best_name = max(results, key=lambda k: results[k]['mean'])
    logger.info('Best model: %s', best_name)
    return best_name, models[best_name], results
