from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationEngine:
    def generate_recommendations(self, data, predictions):
        recs = []
        score = predictions.get("predicted_score", 0)
        if data.get("attendance", 100) < 70:
            recs.append({"area":"Attendance","message":"Increase attendance.","priority":"high"})
        if data.get("study_hours", 50) < 10:
            recs.append({"area":"Study Hours","message":"Aim for 10-15 hours per week.","priority":"high"})
        avg_assign = (data.get("assignment_1", 0) + data.get("assignment_2", 0) + data.get("assignment_3", 0)) / 3
        if avg_assign < 60:
            recs.append({"area":"Assignments","message":"Focus on improving scores.","priority":"medium"})
        if data.get("past_gpa", 4.0) < 2.5:
            recs.append({"area":"Academic Foundation","message":"Review fundamental concepts.","priority":"medium"})
        if score >= 80:
            recs.append({"area":"Excellence","message":"Great performance!","priority":"low"})
        if not recs:
            recs.append({"area":"General","message":"Keep up the good work!","priority":"low"})
        return recs