"""
Customer Churn Prediction API
Serves the trained Gradient Boosting model as a REST endpoint.
Run locally:  uvicorn main:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "churn_model.joblib")
ENCODERS_PATH = os.path.join(BASE, "models", "label_encoders.joblib")
FEATURES_PATH = os.path.join(BASE, "models", "feature_columns.joblib")

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
feature_columns = joblib.load(FEATURES_PATH)


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
                "TotalCharges": 420.75
            }
        }


def build_features(payload: CustomerInput) -> pd.DataFrame:
    row = payload.dict()

    def tenure_bucket(t):
        if t <= 12: return "0-1yr"
        elif t <= 24: return "1-2yr"
        elif t <= 48: return "2-4yr"
        else: return "4yr+"

    row["TenureBucket"] = tenure_bucket(row["tenure"])
    row["AvgMonthlySpend"] = round(row["TotalCharges"] / max(row["tenure"], 1), 2)
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    row["NumAddonServices"] = sum(1 for c in addon_cols if row[c] == "Yes")

    df = pd.DataFrame([row])

    for col, le in encoders.items():
        if col in df.columns and col != "Churn":
            df[col] = le.transform(df[col].astype(str))

    df = df[feature_columns]
    return df


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: CustomerInput):
    X = build_features(payload)
    proba = float(model.predict_proba(X)[0][1])
    prediction = "Yes" if proba >= 0.5 else "No"
    risk = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
    return {
        "churn_prediction": prediction,
        "churn_probability": round(proba, 4),
        "risk_level": risk
    }
