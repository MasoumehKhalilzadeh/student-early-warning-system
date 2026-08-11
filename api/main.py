from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path

# Initialize app
app = FastAPI(
    title="Student Early Warning System",
    description="Predicts whether a student is at risk of failing or withdrawing",
    version="1.0.0"
)

# Load model, scaler and features
models_path = Path(__file__).parent.parent / 'models'

with open(models_path / 'model_binary.pkl', 'rb') as f:
    model = pickle.load(f)

with open(models_path / 'scaler_binary.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(models_path / 'features_binary.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Define input data structure
class StudentData(BaseModel):
    gender_encoded: float
    disability_encoded: float
    age_encoded: float
    education_encoded: float
    num_of_prev_attempts: float
    studied_credits: float
    total_clicks: float
    avg_daily_clicks: float
    active_days: float
    max_clicks_day: float
    early_clicks: float
    avg_score: float
    max_score: float
    min_score: float
    num_assessments: float
    num_late_submissions: float
    late_submission_rate: float
    avg_registration_date: float
    click_consistency: float
    early_engagement_ratio: float
    assessment_completion_rate: float
    score_range: float
    struggle_score: float

# Home route
@app.get("/")
def home():
    return {
        "message": "Student Early Warning System API",
        "version": "1.0.0",
        "status": "running"
    }

# Health check route
@app.get("/health")
def health():
    return {"status": "healthy"}

# Prediction route
@app.post("/predict")
def predict(student: StudentData):
    # Convert input to array
    features = np.array([[
        student.gender_encoded,
        student.disability_encoded,
        student.age_encoded,
        student.education_encoded,
        student.num_of_prev_attempts,
        student.studied_credits,
        student.total_clicks,
        student.avg_daily_clicks,
        student.active_days,
        student.max_clicks_day,
        student.early_clicks,
        student.avg_score,
        student.max_score,
        student.min_score,
        student.num_assessments,
        student.num_late_submissions,
        student.late_submission_rate,
        student.avg_registration_date,
        student.click_consistency,
        student.early_engagement_ratio,
        student.assessment_completion_rate,
        student.score_range,
        student.struggle_score
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    # Format result
    result = "At Risk" if prediction == 1 else "Not At Risk"
    confidence = round(float(max(probability)) * 100, 2)
    at_risk_probability = round(float(probability[1]) * 100, 2)

    return {
        "prediction": result,
        "confidence": f"{confidence}%",
        "at_risk_probability": f"{at_risk_probability}%",
        "recommendation": (
            "Immediate intervention recommended" 
            if prediction == 1 
            else "Student appears to be on track"
        )
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


    if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)