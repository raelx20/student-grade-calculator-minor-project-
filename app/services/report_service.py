import os
from datetime import datetime
from src.utils.constants import BASE_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'prediction_reports')


class ReportService:
    def generate_report(self, data, predictions, recommendations):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = 'report_' + timestamp + '.html'
        filepath = os.path.join(REPORTS_DIR, filename)
        html = self._build_html(data, predictions, recommendations)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info('Report generated: %s', filepath)
        return filepath

    def _build_html(self, data, predictions, recommendations):
        recs = []
        for r in recommendations:
            recs.append('<li><b>' + r['area'] + '</b>: ' + r['message'] + ' (Priority: ' + r['priority'] + ')</li>')
        recs_html = ''.join(recs)
        score = predictions.get('predicted_score', 'N/A')
        grade = predictions.get('grade', 'N/A')
        status = predictions.get('pass_fail', 'N/A')
        return '<!DOCTYPE html><html><head><title>Prediction Report</title></head><body>' + \
            '<h1>Academic Performance Prediction Report</h1>' + \
            '<h2>Predictions</h2><ul>' + \
            '<li>Score: ' + str(score) + '</li>' + \
            '<li>Grade: ' + str(grade) + '</li>' + \
            '<li>Status: ' + str(status) + '</li></ul>' + \
            '<h2>Recommendations</h2><ul>' + recs_html + '</ul>' + \
            '</body></html>'
