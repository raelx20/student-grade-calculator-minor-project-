from src.utils.constants import PASS_THRESHOLD, GRADE_BOUNDARIES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def map_to_grade(score: float) -> str:
    for threshold, grade in GRADE_BOUNDARIES:
        if score >= threshold:
            return grade
    return 'F'


def map_to_grade_batch(scores: list) -> list:
    return [map_to_grade(s) for s in scores]
