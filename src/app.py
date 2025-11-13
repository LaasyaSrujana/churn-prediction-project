from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn
import os

# ============================
# Load saved files
# ============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoders.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "models", "feature_order.joblib")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = joblib.load(FEATURE_ORDER_PATH)

# ============================
# FastAPI App
# ============================

app = FastAPI(title="Customer Churn Prediction API")

# ============================
# Input Schema
# ============================

class CustomerData(BaseModel):
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


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running."}


@app.post("/predict")
def predict_churn(data: CustomerData):

    # Convert input to DataFrame
    df = pd.DataFrame([data.model_dump()])

    # ============================
    # Encode categorical columns
    # ============================
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except:
                df[col] = 0  # unseen categories → safe fallback

    # ============================
    # REORDER COLUMNS EXACTLY AS TRAINING
    # ============================
    df = df.reindex(columns=feature_order)

    # ============================
    # Scale data
    # ============================
    df_scaled = scaler.transform(df)

    # ============================
    # Prediction
    # ============================
    prediction = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0][1]

    result = "Churn" if prediction == 1 else "Not Churn"

    return {
        "prediction": result,
        "probability": round(float(proba), 2)
    }


# ============================
# Run App
# ============================

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
