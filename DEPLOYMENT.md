# Deployment Guide

**Production-Ready MLOps System**

---

## Quick Start (Main Branch)

### 1. **Clone & Setup**

```bash
git clone <repo-url>
cd MLOps
git checkout main  # Deployment-ready branch
```

### 2. **Install Dependencies**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. **Verify Data & Model**

```bash
# Data versioning
dvc pull

# Check MLflow registry
mlflow ui
# Visit http://localhost:5000
# Verify: ClinicalTrialDropout_dropout_v3_causal exists
```

### 4. **Start API Server**

```bash
python api/main.py
```

**Server starts on:** `http://localhost:8000`

**Interactive docs:** `http://localhost:8000/docs`

---

## What's in Main Branch

### ✅ **Deployable Code**

```
MLOps/
├── api/                    # FastAPI application
│   ├── main.py            # API server
│   ├── config.py          # Configuration (no hardcoding)
│   ├── prediction_logger.py  # Server-side logging
│   └── test_api.py        # API tests
├── src/
│   ├── core/              # Production modules
│   │   ├── ingest.py      # Data loading
│   │   ├── features.py    # Feature engineering
│   │   ├── preprocess.py  # Preprocessing pipeline
│   │   └── train.py       # Model comparison (for reference)
│   └── utils/             # Utilities
├── data/                  # DVC-tracked data
├── docs/                  # Knowledge base
├── requirements.txt       # Dependencies
└── .dvc/                  # Data version control
```

### ❌ **Not in Main (in Research Branch)**

- `src/experiments/` - Threshold tuning experiments
- `scripts/tag_mlflow_runs.py` - MLflow management tools
- Jupyter notebooks
- Ad-hoc analysis scripts

**Why?** Enterprise separation: production code stays clean.

---

## Configuration

### Environment Variables

Create `.env` file (optional):

```bash
MODEL_VERSION=v3_causal
MODEL_STAGE=production
DECISION_THRESHOLD=0.20
API_PORT=8000
LOG_LEVEL=INFO
```

Or use defaults from `api/config.py`.

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Make Prediction

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

---

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY src/ src/
COPY data/ data/
COPY mlflow.db mlflow.db

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

**AWS Lambda:**
- Use Mangum adapter for FastAPI
- Deploy with API Gateway

**Google Cloud Run:**
```bash
gcloud run deploy dropout-prediction \
  --source . \
  --platform managed \
  --region us-central1
```

**Azure App Service:**
```bash
az webapp up --name dropout-prediction --runtime "PYTHON:3.10"
```

---

## Monitoring & Logging

### Prediction Logs

**Location:** `logs/predictions.jsonl`

**Format:** JSON Lines (one prediction per line)

**Example:**
```json
{"session_id": "...", "timestamp": "...", "model_version": "v3_causal", "decision_threshold": 0.2, "patient_id": "P-1234", "prediction": 1, "probability": 0.78, "risk_level": "High", "latency_ms": 45.2}
```

### Session Stats

```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "session_id": "...",
  "model_version": "v3_causal",
  "total_predictions": 150,
  "positive_predictions": 85,
  "positive_rate": 0.5667,
  "avg_probability": 0.4521,
  "avg_latency_ms": 42.15,
  "threshold": 0.2
}
```

---

## Model Information

### Selected Model: **LightGBM**

**Why?**
- Highest recall: 82.86% (catches 83% of dropouts)
- Balanced F1 score: 0.6615
- Optimized for clinical retention interventions

**Comparison conducted in research branch:**
- LightGBM vs XGBoost vs Logistic Regression
- See: `research` → `src/experiments/`

### Decision Threshold: **0.20**

**Why lower than default 0.50?**
- Cost-sensitive decision making
- False alarms cheaper than missed dropouts
- Enables early intervention

**Tuning conducted in research branch:**
- Tested thresholds: 0.20, 0.25, 0.30, ..., 0.60
- See: `research` → `src/experiments/threshold_tuning.py`

---

## Documentation

**Knowledge Base (in main branch):**

1. `docs/02_ARCHITECTURE.md` - System design
2. `docs/03_DATA.md` - Data engineering
3. `docs/04_ML_MODEL.md` - Model development
4. `docs/05_MODEL_TUNING.md` - Threshold optimization
5. `docs/MODEL_IO_SPEC.md` - API input/output spec
6. `docs/PREDICTION_LOGGING.md` - Logging architecture
7. `docs/BRANCH_STRATEGY.md` - Production vs research separation

**All docs reference research branch for experiment details.**

---

## Research & Experiments

### Access Research Branch

```bash
git checkout research
```

**Contains:**
- Threshold tuning experiments
- Model comparison scripts
- MLflow tagging utilities
- All experimental code

**Preserved for:**
- Audit trail
- Reproducibility
- Knowledge transfer

### Return to Production

```bash
git checkout main
```

---

## Testing

### API Tests

```bash
python api/test_api.py
```

### Manual Testing

```bash
# High-risk patient (expect dropout=1)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P-TEST-001", "age": 72, "gender": "Female", "treatment_group": "Placebo", "trial_phase": "Phase III", "days_in_trial": 150, "visits_completed": 2, "last_visit_day": 90, "adverse_events": 8}'

# Low-risk patient (expect dropout=0)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P-TEST-002", "age": 45, "gender": "Male", "treatment_group": "Active", "trial_phase": "Phase I", "days_in_trial": 60, "visits_completed": 2, "last_visit_day": 55, "adverse_events": 1}'
```

---

## Troubleshooting

### Error: "Model not found in MLflow"

**Solution:**
```bash
# Train model (creates MLflow registry entry)
python src/core/train.py
```

### Error: "Preprocessor not found"

**Solution:**
```bash
# Generate preprocessor
python src/core/preprocess.py
```

### Error: "No module named 'config'"

**Solution:**
```bash
# Run from project root, not api/ directory
cd MLOps  # Go to root
python api/main.py  # NOT: cd api && python main.py
```

---

## Performance

**Expected metrics:**
- Latency: ~40-50ms per prediction
- Throughput: ~20-25 predictions/second (single worker)
- Memory: ~200MB (model + API)

**Scale with Gunicorn:**
```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## Support & Contribution

### Issues

Report issues to the research team with:
- Prediction logs
- Input data (anonymized)
- Error messages
- Session stats

### Feature Requests

Experiments go in `research` branch first, then graduate to `main` if validated.

---

**Last Updated:** 2025-12-28  
**Branch:** main (production-ready)  
**Model:** LightGBM v3_causal  
**Threshold:** 0.20  
**Status:** ✅ Ready for Deployment
