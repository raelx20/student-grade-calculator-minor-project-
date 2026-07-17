from flask import Blueprint, render_template, request, jsonify
from app.services.prediction_service import PredictionService
from app.services.recommendation_service import RecommendationService

prediction_bp = Blueprint("prediction", __name__)
pred_service = PredictionService()
rec_service = RecommendationService()


@prediction_bp.route("/")
def index():
    return render_template("index.html")


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    data = {
        "assignment_1": float(request.form.get("assignment_1", 0)),
        "assignment_2": float(request.form.get("assignment_2", 0)),
        "assignment_3": float(request.form.get("assignment_3", 0)),
        "attendance": float(request.form.get("attendance", 0)),
        "study_hours": float(request.form.get("study_hours", 0)),
        "past_gpa": float(request.form.get("past_gpa", 0)),
    }
    predictions = pred_service.predict(data)
    recommendations = rec_service.get_recommendations(data, predictions)
    return render_template("result.html", predictions=predictions, recommendations=recommendations, data=data)


@prediction_bp.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    predictions = pred_service.predict(data)
    recommendations = rec_service.get_recommendations(data, predictions)
    return jsonify({"predictions": predictions, "recommendations": recommendations})