import joblib
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def train_xgb(X, y, params):
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model

def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float((np.mean(abs((y_true - y_pred) / (y_true.clip(min=1e-6)))))*100)
    return {'R2': r2, 'RMSE': rmse, 'MAPE': mape}
