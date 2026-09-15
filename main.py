"""
FastAPI backend for Student Mental Health predictor.

Loads a trained scikit-learn pipeline (Gradient Boosting + preprocessing)
and exposes a single POST /predict endpoint. The frontend sends raw student
data, the backend builds interaction features, and the model returns a
mental health score from 0 to 10.
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

# ---------------------------------------------------------------------------
# Load the trained pipeline (preprocessing + model in one .pkl file)
# ---------------------------------------------------------------------------
model = joblib.load('Mental_Health_Model.pkl')

# Top 10 countries by frequency in the dataset — everything else becomes "Other"
# to keep the model from choking on 111 unique country values
top_countries = ['India', 'USA', 'Canada', 'Australia', 'UK',
                 'Germany', 'Mexico', 'Turkey', 'France']

app = FastAPI(title="Student Mental Health API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class StudentData(BaseModel):
    """
    What the frontend sends for each prediction.
    Pydantic validates everything automatically — if someone sends
    age=-5 or stress_level="banana", it rejects the request before
    it reaches the model.
    """
    age                     : int = Field(..., ge=10, le=100)
    gender                  : Literal['Male', 'Female']
    country                 : str
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat',
                                      'Twitter', 'YouTube', 'TikTok', 'LINE',
                                      'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Medium', 'Low', 'Very High', 'High']


class PredictionResponse(BaseModel):
    """What we send back — the score rounded to 2 decimals."""
    predicted_mental_health_score: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get('/')
def greet():
    return {'message': 'Student Mental Health API — send POST to /predict'}


@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    """
    Takes raw student data, groups the country, computes 5 interaction
    features the model was trained on, and returns the prediction.

    Interaction features added here MUST match what was used during
    training (see ML_Project.ipynb).
    """
    country_group = data.country if data.country in top_countries else "Other"

    sleep  = max(data.sleep_hours_per_night, 0.1)
    screen = data.avg_daily_usage_hours
    study  = max(data.study_hours, 0.1)

    input_row = pd.DataFrame([{
        'Age':                    data.age,
        'Gender':                 data.gender,
        'Grouped_country':        country_group,
        'Academic_Level':         data.academic_level,
        'Most_Used_Platform':     data.most_used_platform,
        'Purpose_Of_Use':         data.purpose_of_use,
        'Avg_Daily_Usage_Hours':  screen,
        'Daily_Unlocks':          data.daily_unlocks,
        'Study_Hours':            data.study_hours,
        'Physical_Activity_Hours': data.physical_activity_hours,
        'Sleep_Hours_Per_Night':  data.sleep_hours_per_night,
        'Stress_Level':           data.stress_level,
        # Interaction features
        'Screen_Sleep_Ratio':     screen / sleep,
        'Screen_Study_Ratio':     screen / study,
        'Unlocks_Per_Hour':       data.daily_unlocks / max(screen, 0.1),
        'Lifestyle_Balance':      (data.study_hours + data.physical_activity_hours) / max(screen, 0.1),
        'Sleep_Activity_Balance': sleep + data.physical_activity_hours,
    }])

    prediction = model.predict(input_row)[0]
    return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))
