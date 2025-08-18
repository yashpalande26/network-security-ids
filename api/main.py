from fastapi import FastAPI
import joblib
import pandas as pd
import json
from typing import List, Dict, Any
from fastapi import HTTPException


# Create FastAPI instance
app = FastAPI(title="Network Intrusion Detection API")

# Paths to your saved model + features
MODEL_PATH = "models/random_forest_pipeline.pkl"
FEATURES_PATH = "models/features.json"

# Load model and feature list at startup
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_list = json.load(f)

@app.get("/")
def home():
    return {"message": "Network IDS API is running!"}

@app.post("/predict")
def predict(data: dict):
    # Convert incoming data to DataFrame
    df = pd.DataFrame([data])

    # Ensure correct column order
    df = df[feature_list]

    # Predict
    prediction = model.predict(df)[0]
    return {"prediction": prediction}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_batch")
def predict_batch(items: List[Dict[str, Any]]):
    if not isinstance(items, list) or len(items) == 0:
        raise HTTPException(status_code=422, detail="Request body must be a non-empty JSON list.")

    df = pd.DataFrame(items)

    # Ensure all required features are present
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required feature(s): {missing}"
        )

    # Enforce column order
    df = df[feature_list]

    # Predict
    preds = model.predict(df).tolist()

    # Optional: include probability of DDoS if available
    proba = None
    if hasattr(model, "predict_proba"):
        # assumes class order ["BENIGN","DDoS"]; if reversed, you can inspect model.classes_
        proba = model.predict_proba(df)[:, 1].tolist()  # probability of class 1 (DDoS)

    # Build response
    if proba is not None:
        results = [{"prediction": p, "prob_ddos": round(s, 4)} for p, s in zip(preds, proba)]
        return {"results": results}
    else:
        return {"predictions": preds}
