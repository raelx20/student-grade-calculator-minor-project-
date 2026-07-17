from flask import Blueprint, request, jsonify
from app.services.recommendation_service import RecommendationService

recommendation_bp = Blueprint("recommendation", __name__)
rec_service = RecommendationService()


@recommendation_bp.route("/api/recommendations", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    student_data = data.get("student_data", {})
    predictions = data.get("predictions", {})
    recommendations = rec_service.get_recommendations(student_data, predictions)
    return jsonify({"recommendations": recommendations})