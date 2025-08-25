from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
import math
import io

# ---------------- App ----------------
app = FastAPI(title="Network Intrusion Detection API")

# ---------------- Paths ----------------
MODEL_PATH = "models/random_forest_pipeline.pkl"
FEATURES_PATH = "models/features.json"

# ---------------- Load model & schema at startup ----------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

try:
    with open(FEATURES_PATH, "r") as f:
        FEATURE_NAMES: List[str] = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load feature schema from {FEATURES_PATH}: {e}")

# ---------------- Sanitizers ----------------
def _sanitize_value(v: Any) -> float:
    """Coerce to float; replace NaN/±inf with 0; clip extreme magnitudes."""
    try:
        v = float(v)
    except Exception:
        return 0.0
    if not math.isfinite(v):  # NaN, +inf, -inf
        return 0.0
    if v > 1e12:
        return 1e12
    if v < -1e12:
        return -1e12
    return v

def _sanitize_record(rec: Dict[str, Any]) -> Dict[str, float]:
    """Align a single record to model features and sanitize values."""
    return {name: _sanitize_value(rec.get(name, 0.0)) for name in FEATURE_NAMES}

def _json_safe(obj: Any):
    """Recursively convert to JSON-safe types (no NaN/inf; numpy -> Python)."""
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else 0.0
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    return obj

def _predict_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Run prediction on df that already matches FEATURE_NAMES; return JSON-safe dict."""
    preds = model.predict(df)

    # Try to include probability of DDoS if available
    prob_vals = None
    try:
        if hasattr(model, "predict_proba"):
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                clf = model.named_steps.get("clf")
                classes = getattr(clf, "classes_", None) if clf is not None else None

            ddos_idx = 1
            if classes is not None:
                for i, c in enumerate(classes):
                    if str(c).lower() == "ddos":
                        ddos_idx = i
                        break
            prob_vals = model.predict_proba(df)[:, ddos_idx]
    except Exception:
        prob_vals = None  # ignore probability if anything fails

    if prob_vals is not None:
        results = [{"prediction": str(p), "prob_ddos": float(pr)} for p, pr in zip(preds, prob_vals)]
        return _json_safe({"results": results})

    return _json_safe({"predictions": [str(p) for p in preds]})

# ---------------- Endpoints ----------------
@app.get("/")
def home():
    return {"message": "Network IDS API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/schema")
def schema():
    """Return the ordered feature list expected by the model."""
    return {"features": FEATURE_NAMES}

@app.post("/predict")
def predict(data: Dict[str, Any]):
    """Single-record prediction. Missing features are auto-filled with 0."""
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")
    clean = _sanitize_record(data)
    df = pd.DataFrame([clean], columns=FEATURE_NAMES)
    return _predict_df(df)

@app.post("/predict_batch")
def predict_batch(items: List[Dict[str, Any]]):
    """Batch prediction via JSON list. Missing features -> 0."""
    if not isinstance(items, list) or len(items) == 0:
        raise HTTPException(status_code=422, detail="Request body must be a non-empty JSON list.")
    cleaned = [_sanitize_record(rec) for rec in items]
    df = pd.DataFrame(cleaned, columns=FEATURE_NAMES)
    return _predict_df(df)

@app.post("/predict_batch_file")
async def predict_batch_file(file: UploadFile = File(...)):
    """
    Accept a CSV upload, normalize headers (strip spaces), align to FEATURE_NAMES,
    coerce to numeric, replace NaN/inf with 0, then predict.
    """
    try:
        raw = await file.read()
        # Read CSV into DataFrame
        df = pd.read_csv(io.BytesIO(raw))

        # 1) Normalize headers (strip leading/trailing spaces)
        df.columns = df.columns.astype(str).str.strip()

        # 2) Build aligned frame in the exact expected order
        aligned = pd.DataFrame({name: pd.to_numeric(df.get(name), errors='coerce') for name in FEATURE_NAMES})

        # 3) Replace NaN/±inf with 0 (JSON-safe & model-safe)
        aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 4) Predict using the shared helper
        return _predict_df(aligned)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {e}")
