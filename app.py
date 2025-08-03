# filename: app.py

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import os

# --- FastAPI Setup ---
app = FastAPI(title="Ghost Session Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Artifacts ---
model = joblib.load("models/isolation_forest_behavioral.pkl")
scaler = joblib.load("models/feature_scaler.pkl")
expected_columns = joblib.load("models/feature_columns.pkl")
frequency_encodings = joblib.load("models/frequency_encodings.pkl")

# --- Input Schema ---
class SessionInput(BaseModel):
    user_id: str
    device_os: str
    login_hour: float
    typing_speed_cpm: float
    nav_path: str
    nav_path_depth: float
    ip_country: str
    session_duration_sec: float
    mouse_movement_rate: float
    device_id: str
    ip_consistency_score: float
    login_day_of_week: str
    geo_distance_from_usual: float
    browser_language: str
    failed_login_attempts_last_24h: int
    is_vpn_detected: int
    recent_device_change: int

class SessionBatchInput(BaseModel):
    sessions: List[SessionInput]

# --- Risk Mapping ---
def score_to_risk(score):
    if score <= -0.03:
        return "High"
    elif score <= -0.005:
        return "Medium"
    elif score <= 0.005:
        return "Low"
    else:
        return "None"

# --- Prediction Endpoint ---
@app.post("/predict")
def predict(session_batch: SessionBatchInput):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([s.dict() for s in session_batch.sessions])

        # --- Frequency Encoding ---
        for col in frequency_encodings:
            freq_map = frequency_encodings[col]
            df[f"{col}_encoded"] = df[col].map(freq_map).fillna(0.0)
        df.drop(columns=frequency_encodings.keys(), inplace=True)

        # --- Scaling ---
        if scaler:
            scaled_cols = [col for col in df.columns if col not in ['user_id']]
            df[scaled_cols] = scaler.transform(df[scaled_cols])

        # --- Add placeholder behavioral features ---
        if 'user_similarity_score' not in df.columns:
            df['user_similarity_score'] = 0.5
        if 'user_deviation_score' not in df.columns:
            df['user_deviation_score'] = 0.5

        # --- Ensure all expected columns exist ---
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0.0
        df = df[expected_columns]  # reorder

        # --- Predict ---
        scores = model.decision_function(df)
        risks = [score_to_risk(s) for s in scores]
        predicted_labels = [-1 if r == "High" else 1 for r in risks]

        return {
            "results": [
                {
                    "score": float(s),
                    "risk_level": r,
                    "predicted_label": int(l)
                }
                for s, r, l in zip(scores, risks, predicted_labels)
            ]
        }

    except Exception as e:
        print("Error during prediction:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# --- Run ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


