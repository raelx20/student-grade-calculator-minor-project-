import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.constants import PASS_THRESHOLD, GRADE_BOUNDARIES, CGPA_MIN, CGPA_MAX


def generate_student_data(n_students=800, seed=42):
    np.random.seed(seed)

    assignment_1 = np.clip(np.random.normal(65, 15, n_students), 0, 100).round(1)
    assignment_2 = np.clip(assignment_1 + np.random.normal(2, 8, n_students), 0, 100).round(1)
    assignment_3 = np.clip(assignment_2 + np.random.normal(1, 6, n_students), 0, 100).round(1)
    attendance = np.clip(np.random.normal(75, 15, n_students), 0, 100).round(1)
    study_hours = np.clip(np.random.normal(12, 6, n_students), 0, 50).round(1)
    # CGPA on 0-10 scale
    past_gpa = np.clip(np.random.normal(7.0, 1.5, n_students), CGPA_MIN, CGPA_MAX).round(2)

    noise = np.random.normal(0, 5, n_students)
    final_score = (
        0.25 * assignment_1 +
        0.25 * assignment_2 +
        0.30 * assignment_3 +
        0.10 * attendance +
        0.05 * study_hours * 2 +
        0.05 * (past_gpa / CGPA_MAX) * 100 +
        noise
    )
    final_score = np.clip(final_score, 0, 100).round(1)

    pass_fail = (final_score >= PASS_THRESHOLD).astype(int)

    def score_to_grade(score):
        for threshold, grade in GRADE_BOUNDARIES:
            if score >= threshold:
                return grade
        return 'F'

    grade = np.array([score_to_grade(s) for s in final_score])

    df = pd.DataFrame({
        'student_id': range(1, n_students + 1),
        'assignment_1': assignment_1,
        'assignment_2': assignment_2,
        'assignment_3': assignment_3,
        'attendance': attendance,
        'study_hours': study_hours,
        'past_gpa': past_gpa,
        'final_score': final_score,
        'pass_fail': pass_fail,
        'grade': grade,
    })

    return df


if __name__ == '__main__':
    df = generate_student_data()
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'students.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print('Generated ' + str(len(df)) + ' students -> ' + output_path)
    print('Grade distribution:')
    print(df['grade'].value_counts().sort_index().to_string())
