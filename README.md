# Academic Performance Predictor

ML pipeline that predicts student final grades (A+ through F) and pass/fail status based on assignment scores, attendance, study hours, and past performance.

## Setup
pip install -r requirements.txt

## Generate Data
python src/utils/generate_data.py

## Train Models
python -m src.training.train_score
python -m src.training.train_pass

## Run App
python app/main.py
Open http://localhost:5000 in your browser.

## Run Tests
pytest tests/ -v
