# SI Prediction & Anomaly Detection FastAPI Service

## Train models
```bash
python -m src.train
```

## Run service
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8080
```

## Endpoints
- GET /health
- POST /predict

