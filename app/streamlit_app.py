import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Config ----------------
DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")
FEATURES_PATH = os.path.join("models", "features.json")
with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

st.set_page_config(page_title="Network IDS – Demo", layout="wide")

# ---------------- Utilities ----------------
def prep_record_for_api(rec: dict) -> dict:
    """Make a single record JSON-safe and aligned to the model features."""
    clean = {}
    for k in FEATURES:
        v = rec.get(k, 0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        clean[k] = float(v)
    return clean

def post_json(url: str, payload: dict | list, timeout: int = 60):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    except Exception as e:
        return 0, str(e)

# ---------------- Sidebar ----------------
st.sidebar.header("Backend")
base_url = st.sidebar.text_input("FastAPI base URL", value=DEFAULT_API)
if st.sidebar.button("Check /health"):
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        st.sidebar.success(r.json())
    except Exception as e:
        st.sidebar.error(f"Health check failed: {e}")

st.title("Network Intrusion Detection – Demo")
st.caption("Streamlit UI over FastAPI. BENIGN vs DDoS. Auto-aligns to model schema.")

tabs = st.tabs(["Single", "Batch CSV"])

# ---------------- Single ----------------
with tabs[0]:
    st.subheader("Single Prediction")
    primary_feats = FEATURES[:10]
    cols = st.columns(3)
    single_values = {}

    for i, feat in enumerate(primary_feats):
        with cols[i % 3]:
            single_values[feat] = st.number_input(feat, value=0.0, step=1.0, format="%.6f")

    with st.expander("Show remaining features"):
        for feat in FEATURES[10:]:
            single_values[feat] = st.number_input(feat, value=0.0, step=1.0, format="%.6f")

    if st.button("Predict", type="primary"):
        payload = prep_record_for_api(single_values)
        code, data = post_json(f"{base_url}/predict", payload)
        if code == 200:
            st.success(data)
        else:
            st.error(f"Error {code}: {data}")

# ---------------- Batch CSV (server reads the file) ----------------
with tabs[1]:
    st.subheader("Batch CSV Prediction (server reads the file)")
    st.caption("Upload a CSV; the API cleans NaN/±inf and predicts. You’ll get a download with all rows + Prediction.")

    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        # Read the uploaded file bytes once, reuse for preview + API
        file_bytes = up.getvalue()

        # Preview (local only)
        try:
            df_preview = pd.read_csv(io.BytesIO(file_bytes), nrows=5)
            st.write("Preview of uploaded CSV:")
            st.dataframe(df_preview, use_container_width=True)
        except Exception as e:
            st.warning(f"Preview failed (encoding/format): {e}")

        if st.button("Run batch prediction"):
            with st.spinner("Predicting…"):
                # Send the file bytes to the API (no JSON float issues)
                files = {"file": ("batch.csv", file_bytes, "text/csv")}
                try:
                    r = requests.post(f"{base_url}/predict_batch_file", files=files, timeout=180)
                    if r.status_code != 200:
                        st.error(f"API error {r.status_code}: {r.text}")
                    else:
                        data = r.json()

                        # Extract predictions from API response
                        if isinstance(data, dict) and "results" in data:
                            # results likely a list of dicts like {"prediction": "...", "prob_ddos": 0.99}
                            pred_list = [str(row.get("prediction", "")) for row in data["results"]]
                        elif isinstance(data, dict) and "predictions" in data:
                            pred_list = [str(p) for p in data["predictions"]]
                        else:
                            st.error("Unexpected API response format.")
                            st.json(data)
                            pred_list = None

                        if pred_list is not None:
                            # Load the FULL CSV locally
                            try:
                                df_full = pd.read_csv(io.BytesIO(file_bytes))
                            except Exception as e:
                                st.error(f"Failed to read full CSV locally: {e}")
                                df_full = None

                            if df_full is not None:
                                # Length check
                                if len(df_full) != len(pred_list):
                                    st.warning(
                                        f"Row mismatch: CSV has {len(df_full)} rows "
                                        f"but API returned {len(pred_list)} predictions. "
                                        "We’ll still display what we can."
                                    )

                                # Add prediction column (trim to the shorter length to be safe)
                                n = min(len(df_full), len(pred_list))
                                df_out = df_full.iloc[:n].copy()
                                df_out["Prediction"] = pred_list[:n]

                                st.success("Batch prediction complete")
                                st.dataframe(df_out.head(50), use_container_width=True)

                                # ---- Download button: all rows + Prediction ----
                                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                                suggested_name = f"predictions_{getattr(up, 'name', 'batch')}"
                                st.download_button(
                                    label="Download predictions CSV",
                                    data=csv_bytes,
                                    file_name=suggested_name,
                                    mime="text/csv",
                                )
                            else:
                                # If we couldn't load CSV locally, at least show predictions
                                st.success("Batch prediction complete (showing predictions only)")
                                st.dataframe(pd.DataFrame({"Prediction": pred_list}).head(50))
                except Exception as e:
                    st.error(f"Request error: {e}")

st.caption(f"Loaded {len(FEATURES)} model features • API: {base_url}")