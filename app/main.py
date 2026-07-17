import os
import sys
from flask import Flask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config import config
from app.api.prediction import prediction_bp
from app.api.recommendation import recommendation_bp
from app.api.health import health_bp


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = 'academic-predictor-secret'
    app.register_blueprint(prediction_bp)
    app.register_blueprint(recommendation_bp)
    app.register_blueprint(health_bp)
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host=config.host, port=config.port, debug=config.debug)
