# Steelmaking SI Prediction & Anomaly Detection

This repository provides an end-to-end solution for predicting the Silicon Index (SI) in steelmaking, detecting anomalies, and suggesting process corrections using machine learning and explainable AI.

## Features

- **Data Preprocessing:** Cleans and imputes missing values using KNN imputer.
- **Feature Engineering:** Adds time-based, lag, and rolling window features.
- **Modeling:** Trains multiple regression models (Linear Regression, Random Forest, XGBoost).
- **Uncertainty Estimation:** Uses quantile regression to estimate prediction intervals.
- **Explainability:** Provides SHAP-based feature importance and root-cause analysis for anomalies.
- **Anomaly Detection:** Detects anomalies in SI predictions using Isolation Forest and residual analysis.
- **Genetic Algorithm Optimizer:** Suggests optimal setpoints for process variables.
- **Streamlit App:** Interactive UI for predictions, anomaly detection, and actionable recommendations.

## Folder Structure

```
.
├── src/
│   ├── si_full_notebook.ipynb   # Main notebook for E2E workflow
│   ├── app.py                   # Streamlit app for UI
│   ├── utils.py                 # Feature preparation and utility functions
│   ├── realtime.py              # Real-time correction suggestion logic
│   └── models/                  # Saved models and artifacts
├── data/
│   └── DataSet.xlsx             # Input data file (not included)
├── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/si_full_submission.git
cd si_full_submission
```

### 2. Prepare Data

- Place your dataset at `data/DataSet.xlsx`.

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

- Open `src/si_full_notebook.ipynb` in Jupyter or VS Code.
- Run all cells to preprocess data, train models, and generate artifacts.

### 5. Launch the Streamlit App

```bash
cd src
streamlit run app.py
```

## Usage

- Enter process variable values in the Streamlit UI.
- Get SI prediction, anomaly detection, and actionable suggestions.
- Optionally, enter actual SI to check for anomalies in real time.

## Artifacts

- Trained models and imputers are saved in the `src/models/` directory:
  - `model_xgb.joblib` — XGBoost regression model
  - `knn_imputer.joblib` — KNN imputer for missing values
  - `iforest_features.joblib` — Isolation Forest for anomaly detection
  - `model_q_low.joblib`, `model_q_high.joblib` — Quantile regression models
  - `anomaly_report.csv`, `anomaly_root_causes.csv` — Anomaly and root-cause reports

## Customization

- Update `BOUNDS` in the genetic algorithm section for your process variables.
- Modify `suggest_corrections` in `realtime.py` for custom recommendations.

## License

MIT License

---

**Contact:**  
For questions or support, please open an issue or contact the