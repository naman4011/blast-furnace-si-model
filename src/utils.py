# utility functions (if needed)
# src/utils.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import KNNImputer
import joblib

# ---------------------------
# Utility functions
# ---------------------------

def get_prev_df(DATA_PATH='src/data/DataSet.xlsx'):
    TIME_COL='Timestamp'
    TARGET='SI'
    df=pd.read_excel(DATA_PATH)
    df[TIME_COL]=pd.to_datetime(df[TIME_COL])
    df=df.sort_values(TIME_COL).reset_index(drop=True)

    df = df.set_index(TIME_COL).sort_index()
    df = df.ffill().bfill()
    for c in df.columns:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    df = df.reset_index()

    NUMERIC_COLS = ['OxEnRa','BlFuPeIn','EnOxFl','CoBlFl','BlMo','BlFuBoGaVo','BlFuBoGaIn','ThCoTe','ToGaPr','EnOxPr','CoBlPr','ToPrDr','HoBlPr','AcBlVe','CoBlTe','HoBlTe','ToTe','BlHu','CoInSeVa','FoSI','HoBl','ToGasP','CoBF']
    NUMERIC_COLS = [c for c in NUMERIC_COLS if c in df.columns]

    def add_time_feats(d):
        d['hour']=d[TIME_COL].dt.hour
        d['dayofweek']=d[TIME_COL].dt.dayofweek
        d['month']=d[TIME_COL].dt.month
        return d

    df = add_time_feats(df)

    def add_lags_rolls(d, cols, lags=(1,2,3), windows=(3,6,12)):
        dd = d.copy()
        for c in cols:
            for L in lags:
                dd[f'{c}_lag{L}'] = dd[c].shift(L)
            for w in windows:
                dd[f'{c}_r{w}_mean'] = dd[c].rolling(window=w, min_periods=1).mean()
                dd[f'{c}_r{w}_std'] = dd[c].rolling(window=w, min_periods=1).std()
        return dd

    df = add_lags_rolls(df, NUMERIC_COLS)
    df = df.dropna().reset_index(drop=True)
    df = df.drop(columns=["SI","Timestamp"], errors="ignore")  # drop target if present
    return df

def add_time_feats(d: pd.DataFrame, time_col="timestamp"):
    d["hour"] = d[time_col].dt.hour
    d["dayofweek"] = d[time_col].dt.dayofweek
    d["month"] = d[time_col].dt.month
    return d

def add_lags_rolls(d, cols, lags=(1,2,3), windows=(3,6,12)):
    dd = d.copy()
    for c in cols:
        for L in lags:
            dd[f'{c}_lag{L}'] = np.nan
        for w in windows:
            dd[f'{c}_r{w}_mean'] = np.nan
            dd[f'{c}_r{w}_std'] = np.nan
    return dd


# ---------------------------
# Build imputing function
# ---------------------------

def prepare_features(user_input: dict, numeric_cols: list, knn_imputer: KNNImputer):
    """
    user_input: dict with user-provided features (subset of numeric_cols)
    df: historical dataframe with full features (already processed with lag + roll)
    numeric_cols: list of base numeric columns
    knn_k: number of neighbors for KNNImputer
    """
    df = get_prev_df()
    
    # Step 1: Create a single-row DataFrame from user input
    current_time = datetime.now()
    input_df = pd.DataFrame([user_input])
    input_df["timestamp"] = pd.to_datetime(current_time)

    # Step 2: Add time features
    input_df = add_time_feats(input_df, time_col="timestamp")

    # Step 3: Add empty lag + rolling columns (will be NaN, filled later)
    input_df =  add_lags_rolls(input_df, numeric_cols)

    # Step 4: Align with df columns
    missing_cols = [c for c in df.columns if c not in input_df.columns]
    for c in missing_cols:
        input_df[c] = np.nan
    input_df = input_df[df.columns]  # reorder

     #Remove time columns before imputing
    input_for_impute = input_df.drop(columns=["Timestamp", "timestamp"], errors="ignore")
    # Step 5: Transform the input row
    input_imputed = knn_imputer.transform(input_for_impute)

    # Step 6: Return as DataFrame
    return pd.DataFrame(input_imputed, columns=df.columns)

