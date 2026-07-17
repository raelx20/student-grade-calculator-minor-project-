import os
from flask import Blueprint, jsonify
from src.utils.constants import SCORE_MODEL, PASS_MODEL

health_bp = Blueprint('health', __name__)


@health_bp.route('/health')
def health_check():
    models_ready = os.path.exists(SCORE_MODEL) and os.path.exists(PASS_MODEL)
    status = 'healthy' if models_ready else 'models_not_found'
    return jsonify({'status': status, 'models_loaded': models_ready})
