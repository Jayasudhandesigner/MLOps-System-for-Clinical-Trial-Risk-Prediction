# API Quick Start Guide

**Running the Dropout Prediction API**

---

## Prerequisites

1. **Model artifacts exist:**
   - `data/processed/preprocessor_dropout_v3_causal.pkl`
   - Model in MLflow registry: `ClinicalTrialDropout_dropout_v3_causal`

2. **Dependencies installed:**
   ```bash
   pip install fastapi uvicorn pydantic mlflow lightgbm requests
   ```

---

## Start the API Server

### Option 1: Run directly

```bash
cd api
python main.py
```

### Option 2: Run with uvicorn

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     🚀 Starting API server...
INFO:     ✅ Loaded preprocessor: data/processed/preprocessor_dropout_v3_causal.pkl
INFO:     ✅ Loaded model from MLflow: models:/ClinicalTrialDropout_dropout_v3_causal/latest
INFO:     ✅ Prediction logger initialized
INFO:     📊 Model: v3_causal | Stage: production | Threshold: 0.20
```

---

## Test the API

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

### 2. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P-1234",
    "age": 65,
    "gender": "Female",
    "treatment_group": "Placebo",
    "trial_phase": "Phase III",
    "days_in_trial": 120,
    "visits_completed": 3,
    "last_visit_day": 105,
    "adverse_events": 4
  }'
```

**Response:**
```json
{
  "patient_id": "P-1234",
  "dropout_prediction": 1,
  "risk_level": "High",
  "recommended_action": "weekly_monitoring"
}
```

### 3. Run Test Suite

```bash
python api/test_api.py
```

### 4. Check Session Stats (Admin)

```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "session_id": "a7f3c8d2-1e4b-4a9c-8f2d-9c3e1b5a7f8e",
  "model_version": "v3_causal",
  "total_predictions": 15,
  "positive_predictions": 8,
  "positive_rate": 0.5333,
  "avg_probability": 0.4521,
  "avg_latency_ms": 42.15,
  "threshold": 0.2
}
```

---

## Interactive API Documentation

FastAPI automatically generates interactive docs:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

You can test all endpoints directly from the browser!

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/stats` | GET | Get session statistics |

---

## Prediction Logs

All predictions are logged to: `logs/predictions.jsonl`

**Example log entry:**
```json
{
  "session_id": "...",
  "timestamp": "2025-12-28T22:12:00Z",
  "model_version": "v3_causal",
  "decision_threshold": 0.2,
  "patient_id": "P-1234",
  "prediction": 1,
  "probability": 0.7834,
  "risk_level": "High",
  "latency_ms": 45.2
}
```

---

## Troubleshooting

### Error: "Model not found"

**Solution:** Train model first or check MLflow registry:
```bash
python src/core/train.py
```

### Error: "Preprocessor not found"

**Solution:** Run preprocessing:
```bash
python src/core/preprocess.py
```

### Error: "ModuleNotFoundError: No module named 'prediction_logger'"

**Solution:** Run from project root, not api/ directory:
```bash
# Wrong
cd api
python main.py

# Correct
python api/main.py
```

Or add api/ to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/api"
uvicorn api.main:app --reload
```

---

## Example Python Client

```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "patient_id": "P-9999",
        "age": 55,
        "gender": "Male",
        "treatment_group": "Active",
        "trial_phase": "Phase II",
        "days_in_trial": 90,
        "visits_completed": 3,
        "last_visit_day": 85,
        "adverse_events": 2
    }
)

result = response.json()
print(f"Prediction: {result['dropout_prediction']}")
print(f"Risk: {result['risk_level']}")
```

---

## Production Deployment

For production, use a production ASGI server:

```bash
# Install gunicorn
pip install gunicorn

# Run with workers
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

---

**Last Updated:** 2025-12-28  
**API Version:** 1.0.0  
**Model Version:** v3_causal
