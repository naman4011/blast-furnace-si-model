# Not used as we are training models in notebook for better visibility
# Just kept for reference

import pandas as pd
import joblib
import json
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from xgboost import XGBRegressor

def train():
    df = pd.read_csv("data/si_data.csv")
    y = df["SI"]
    X = df.drop(columns=["SI","Timestamp"], errors="ignore")
    feature_stats = {c: {"median": float(df[c].median()), "mean": float(df[c].mean()), "std": float(df[c].std())} for c in X.columns}
    # Models
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)
    xgb.fit(X,y)
    q_low = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=200)
    q_high = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=200)
    q_low.fit(X,y); q_high.fit(X,y)
    iforest = IsolationForest(contamination=0.02, random_state=42)
    iforest.fit(X)
    # Save artifacts
    joblib.dump(xgb, "models/model_xgb.joblib")
    joblib.dump(q_low, "models/model_q_low.joblib")
    joblib.dump(q_high, "models/model_q_high.joblib")
    joblib.dump(iforest, "models/iforest_features.joblib")
    with open("models/feature_stats.json","w") as f:
        json.dump(feature_stats,f,indent=2)
    print("Artifacts saved in models/")

if __name__=="__main__":
    train()
