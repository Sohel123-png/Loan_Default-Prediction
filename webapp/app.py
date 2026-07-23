"""
FastAPI Backend — Loan Default Prediction
=============================================
Yeh app do kaam karta hai:
1. '/' par frontend (static/index.html) serve karta hai
2. '/api/predict' par trained ML pipeline se loan-default prediction deta hai

Model kahan se aata hai?
------------------------
Yeh app '../artifacts/best_pipeline.joblib' (best model, jo
'06_model_comparison_ensembles.py' + '07_save_and_predict.py' se banta hai)
load karta hai. Agar file nahi milti to server crash NAHI hota — startup par
ek clear warning print hoti hai aur '/api/predict' call karne par ek
readable error message milta hai (frontend usse achhe se dikhata hai).

Run karne ka tareeka:
    cd webapp
    pip install fastapi uvicorn[standard] joblib pandas numpy scikit-learn xgboost lightgbm catboost
    uvicorn app:app --reload --port 8000

Phir browser me: http://127.0.0.1:8000
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Literal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_pipeline.joblib")
MODEL_NAME_PATH = os.path.join(ARTIFACT_DIR, "best_model_name.txt")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Loan Default Prediction API",
    description="Underwriting API — applicant data leke default-risk predict karta hai.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (startup par ek baar)
# ---------------------------------------------------------------------------
pipeline = None
model_name = "Unknown"

def load_model():
    global pipeline, model_name
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        if os.path.exists(MODEL_NAME_PATH):
            with open(MODEL_NAME_PATH) as f:
                model_name = f.read().strip()
        print(f"[startup] Model loaded successfully: {model_name}")
    else:
        print(f"[startup] WARNING: Model file not found at {MODEL_PATH}. "
              f"'/api/predict' will return an error until you train and save a model "
              f"(run 01 -> 07 scripts).")

@app.on_event("startup")
def on_startup():
    load_model()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    person_income: float = Field(..., gt=0, description="Annual income")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(..., ge=0, le=70, description="Employment length in years")
    loan_intent: Literal[
        "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
        "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    loan_int_rate: float = Field(..., ge=0, le=50, description="Interest rate (%)")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan amount / income ratio")
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60, description="Credit history length in years")

    class Config:
        json_schema_extra = {
            "example": {
                "person_age": 28,
                "person_income": 70000,
                "person_home_ownership": "RENT",
                "person_emp_length": 5,
                "loan_intent": "EDUCATION",
                "loan_grade": "B",
                "loan_amnt": 12000,
                "loan_int_rate": 10.5,
                "loan_percent_income": 0.17,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 4,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    verdict: Literal["APPROVED", "DECLINED"]
    default_probability: float
    risk_level: Literal["Low", "Moderate", "High", "Very High"]
    model_used: str


def risk_bucket(prob: float) -> str:
    if prob < 0.15:
        return "Low"
    elif prob < 0.35:
        return "Moderate"
    elif prob < 0.60:
        return "High"
    return "Very High"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "model_name": model_name,
    }


@app.get("/api/model-info")
def model_info():
    metrics_path = os.path.join(ARTIFACT_DIR, "model_comparison_results.csv")
    metrics = None
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        metrics = df.sort_values(by="Mean ROC AUC", ascending=False).to_dict(orient="records")
    return {
        "model_loaded": pipeline is not None,
        "best_model_name": model_name,
        "comparison": metrics,
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model abhi load nahi hai. Pehle training pipeline (scripts 01-07) "
                "chalao taaki 'artifacts/loan_default_pipeline.joblib' ban sake, "
                "phir server restart karo."
            ),
        )

    data = application.model_dump()

    # FIX ke consistent: training time par 'person_income' drop karke
    # 'person_income_log' feature use hua tha (03_feature_engineering.py) —
    # isliye yahan bhi wahi transformation apply karna zaroori hai.
    row = pd.DataFrame([data])
    row["person_income_log"] = np.log1p(row["person_income"])
    row = row.drop(columns=["person_income"])

    try:
        pred = int(pipeline.predict(row)[0])
        prob = float(pipeline.predict_proba(row)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        prediction=pred,
        verdict="DECLINED" if pred == 1 else "APPROVED",
        default_probability=round(prob, 4),
        risk_level=risk_bucket(prob),
        model_used=model_name,
    )


# ---------------------------------------------------------------------------
# Serve frontend (static files) — yeh line sabse last me honi chahiye
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
