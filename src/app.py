from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import shap
import json
from realtime import suggest_corrections
from utils import feature_mapping, prepare_features
import uvicorn
import streamlit as st
import pandas as pd
from datetime import datetime

app = FastAPI(title="SI Prediction & Anomaly Detection API")

# Load artifacts
model = joblib.load("src/models/model_xgb.joblib")
q_low = joblib.load("src/models/model_q_low.joblib")
q_high = joblib.load("src/models/model_q_high.joblib")
iforest = joblib.load("src/models/iforest_features.joblib")
knn_imputer = joblib.load("src/models/knn_imputer.joblib")
with open("src/models/feature_stats.json") as f:
    feature_stats = json.load(f)

NUMERIC_COLS = ['OxEnRa','BlFuPeIn','EnOxFl','CoBlFl','BlMo','BlFuBoGaVo','BlFuBoGaIn',
                'ThCoTe','ToGaPr','EnOxPr','CoBlPr','ToPrDr','HoBlPr','AcBlVe','CoBlTe',
                'HoBlTe','ToTe','BlHu','CoInSeVa','FoSI','HoBl','ToGasP','CoBF']

# Feature mapping (human → dataset variable name)
feature_mapping = {
    "Timestamp": "Timestamp",
    "Oxygen enrichment rate": "OxEnRa",
    "Blast furnace permeability index": "BlFuPeIn",
    "Enriching oxygen flow": "EnOxFl",
    "Cold blast flow": "CoBlFl",
    "Blast momentum": "BlMo",
    "Blast furnace bosh gas volume": "BlFuBoGaVo",
    "Blast furnace bosh gas index": "BlFuBoGaIn",
    "Theoretical combustion temperature": "ThCoTe",
    "Top gas pressure": "ToGaPr",
    "Enriching oxygen pressure": "EnOxPr",
    "Cold blast pressure": "CoBlPr",
    "Total pressure drop": "ToPrDr",
    "Hot blast pressure": "HoBlPr",
    "Actual blast velocity": "AcBlVe",
    "Cold blast temperature": "CoBlTe",
    "Hot blast temperature": "HoBlTe",
    "Top temperature": "ToTe",
    "Blast humidity": "BlHu",
    "Coal injection set value": "CoInSeVa",
    "Fomer SI": "FoSI",
    "SI": "SI"
}
capp = FastAPI()

class InputData(BaseModel):
    features: dict  # key: human name, value: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    
    df = prepare_features(data, NUMERIC_COLS, knn_imputer)

    # Predict SI
    si_pred = model.predict(df)[0]

    # Anomaly detection
    anomaly_flag = int(iforest.predict(df)[0] == -1)

    # Threshold-based decision
    is_outlier = not (q_low <= si_pred <= q_high)

    return {
        "predicted_SI": float(si_pred),
        "anomaly_flag": anomaly_flag,
        "threshold_outlier": is_outlier
    }

# ===============================
# Streamlit UI
# ===============================
def streamlit_ui():
    st.title("Steelmaking SI Prediction & Anomaly Detection")

    user_inputs = {}
    for human_name in feature_mapping.keys():
        if human_name in ["Timestamp", "SI"]:
            continue
        val = st.number_input(f"{human_name}", value=np.nan)
        user_inputs[human_name] = val

    actual_si = st.number_input("Enter Actual SI (optional, for anomaly detection)", value=0.0)


    if st.button("Predict"):
        
        mapped_inputs = {feature_mapping[k]: v for k, v in user_inputs.items()}
        df = prepare_features(mapped_inputs, NUMERIC_COLS, knn_imputer)
        si_pred = model.predict(df)[0]
        # Get actual SI from user (optional input)
        
        q_low_val = q_low.predict(df)[0]
        q_high_val = q_high.predict(df)[0]
        is_outlier = not (q_low_val <= si_pred <= q_high_val)

        # Residual anomaly detection
        residual, anomaly_flag = None, None
        if actual_si != 0.0:
            residual = actual_si - si_pred
            anomaly_flag = int(iforest.predict([[residual]])[0] == -1)

        st.subheader("Results")
        st.write(f"**Predicted SI:** {si_pred:.2f}")
        st.write(f"**Threshold Outlier:** {'Yes' if is_outlier else 'No'}")

        if residual is not None:
            st.write(f"**Residual (Actual - Predicted):** {residual:.2f}")
            st.write(f"**Residual Anomaly (IsolationForest):** {'Yes' if anomaly_flag else 'No'}")
        else:
            st.write("**Residual Anomaly:** Skipped (no Actual SI provided)")
        
        # SHAP explanations
        explainer = shap.Explainer(model)
        shap_values = explainer(df)
        st.subheader("Feature Contributions (SHAP values)")
        shap.initjs()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(bbox_inches='tight')
        
        # Suggestions
        top_features = {c: abs(shap_values.values[0][i]) for i, c in enumerate(df.columns)}
        top_features = dict(sorted(top_features.items(), key=lambda item: item[1], reverse=True)[:3])
        action_block = suggest_corrections(si_pred, q_low_val, q_high_val, top_features)
        suggestions = action_block.get("recommendations", [])

        st.subheader("Correction Suggestions")
        for key, value in action_block.items():
            if key == "recommendations":
                if suggestions:
                    st.write("**Recommendations:**")
                    for rec in suggestions:
                        st.write(f"- {rec}")
                else:
                    st.write("No recommendations available.")
            else:
                st.write(f"**{key.capitalize()}:** {value}")
                
if __name__ == "__main__":
    import sys
    if "fastapi" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        streamlit_ui()
