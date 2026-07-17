import os

# Column names
COLUMNS = {
    'student_id': 'student_id',
    'assignment_1': 'assignment_1',
    'assignment_2': 'assignment_2',
    'assignment_3': 'assignment_3',
    'attendance': 'attendance',
    'study_hours': 'study_hours',
    'past_gpa': 'past_gpa',
    'final_score': 'final_score',
    'pass_fail': 'pass_fail',
    'grade': 'grade',
}

FEATURE_COLUMNS = ['assignment_1', 'assignment_2', 'assignment_3', 'attendance', 'study_hours', 'past_gpa']
TARGET_SCORE = 'final_score'
TARGET_PASS = 'pass_fail'
TARGET_GRADE = 'grade'

# Grade boundaries (percentage-based)
GRADE_BOUNDARIES = [
    (90, 'A+'),
    (80, 'A'),
    (70, 'B+'),
    (60, 'B'),
    (50, 'C'),
    (40, 'D'),
    (0, 'F'),
]

PASS_THRESHOLD = 40

# CGPA range (0-10 scale)
CGPA_MIN = 0
CGPA_MAX = 10

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'students.csv')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'processed_students.csv')
DATA_SAMPLE = os.path.join(BASE_DIR, 'data', 'sample', 'sample_input.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Model paths
SCORE_MODEL = os.path.join(MODELS_DIR, 'score_model.pkl')
PASS_MODEL = os.path.join(MODELS_DIR, 'pass_model.pkl')
GRADE_MODEL = os.path.join(MODELS_DIR, 'grade_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'encoder.pkl')

# Training
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
