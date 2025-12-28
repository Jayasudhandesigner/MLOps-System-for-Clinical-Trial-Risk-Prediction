# Clinical Trial Dropout Prediction - Production MLOps System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/Data-DVC-945DD6.svg)](https://dvc.org/)

**Production-ready machine learning system for predicting patient dropout risk in clinical trials with cost-sensitive decision optimization.**

🎯 **Catches 83% of dropouts** | 💰 **$258K savings per 1000 patients** | 🚀 **Enterprise-grade deployment**

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Jayasudhandesigner/MLOps-System-for-Clinical-Trial-Risk-Prediction.git
cd MLOps-System-for-Clinical-Trial-Risk-Prediction
pip install -r requirements.txt
```

### 2. Start API Server
```bash
python api/main.py
```

**API:** http://localhost:8000  
**Docs:** http://localhost:8000/docs

### 3. Make Prediction
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

## 🎯 Key Results

### Model Performance (LightGBM with Threshold 0.35)

| Metric | Value | Clinical Impact |
|--------|-------|-----------------|
| **Recall** | **72.38%** | ✅ Catches 72% of all dropouts |
| **Precision** | 57.58% | 58% of flagged patients drop out |
| **F1 Score** | 0.6414 | Balanced performance |
| **ROI** | **Optimized cost-benefit** | Reduced false alarms |

**Balanced Threshold:**
- Threshold: **0.35** (balanced between recall and precision)
- Previous aggressive: 0.20 (82.86% recall, many false alarms)
- More sustainable for production use


---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Production Server                 │
│  • Input validation (Pydantic)                               │
│  • Feature engineering (rate-based + domain risk)            │
│  • Model inference (MLflow registry)                         │
│  • Server-side logging (audit trail)                         │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LightGBM Model (v3_causal)                │
│  • Trained on causal features                                │
│  • Decision threshold: 0.35 (balanced)                        │
│  • Stored in MLflow Model Registry                           │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Prediction Output + Logging               │
│  • User: Clean response (no metadata)                        │
│  • Server: Full audit trail (JSONL logs)                     │
└─────────────────────────────────────────────────────────────┘
```

**Technology Stack:**
- **API:** FastAPI + Uvicorn
- **ML:** LightGBM, XGBoost, Scikit-learn
- **MLOps:** MLflow (tracking + registry)
- **Data:** DVC (version control)
- **Validation:** Pydantic
- **Logging:** JSON Lines format

---

## 📁 Repository Structure

### **Main Branch** (Production-Ready)
```
MLOps/
├── api/                         # 🚀 FastAPI Application
│   ├── main.py                  # API server
│   ├── config.py                # Configuration (no hardcoding!)
│   ├── prediction_logger.py    # Server-side logging
│   ├── test_api.py              # API tests
│   └── README.md                # API quick start
│
├── src/core/                    # 🔧 Production Modules
│   ├── ingest.py                # Data loading & validation
│   ├── features.py              # Feature engineering
│   ├── preprocess.py            # Preprocessing pipeline
│   └── train.py                 # Model comparison
│
├── data/                        # 📊 Data (DVC tracked)
│   ├── raw/                     # Raw CSV
│   ├── processed/               # Preprocessed + artifacts
│   └── synthetic_data_causal.py # Data generation
│
├── docs/                        # 📚 Knowledge Base
│   ├── 02_ARCHITECTURE.md       # System design
│   ├── 03_DATA.md               # Data engineering
│   ├── 04_ML_MODEL.md           # Model development
│   ├── 05_MODEL_TUNING.md       # Threshold optimization
│   ├── MODEL_IO_SPEC.md         # API specification
│   ├── PREDICTION_LOGGING.md    # Logging architecture
│   └── BRANCH_STRATEGY.md       # Branch organization
│
├── DEPLOYMENT.md                # 🚢 Deployment guide
├── PROJECT_COMPLETE.md          # 📋 Completion summary
├── requirements.txt             # Dependencies
└── .dvc/                        # Data version control
```

### **Research Branch** (Experiments)
```
+ src/experiments/               # Threshold tuning, analysis
+ scripts/tag_mlflow_runs.py     # MLflow utilities
+ Full experiment history        # Preserved for audit
```

**Branch Strategy:** Enterprise-grade separation (production/research)  
See: [`docs/BRANCH_STRATEGY.md`](docs/BRANCH_STRATEGY.md)

---

## 🤖 Model Selection

### **Why LightGBM?**

Compared three models on causal features:

| Model | Recall | F1 Score | Decision |
|-------|--------|----------|----------|
| **LightGBM** | **0.8286** ✅ | **0.6615** ✅ | **Selected** |
| XGBoost | 0.5524 | 0.5771 | ❌ |
| Logistic Regression | 0.4476 | 0.5402 | ❌ |

**Rationale:** Maximize recall for early dropout detection.

**Experiments:** Conducted in `research` branch  
**Documentation:** [`docs/04_ML_MODEL.md`](docs/04_ML_MODEL.md)

---

## 🎚️ Threshold Optimization

### **Cost-Sensitive Decision Policy**

**Problem:** Default threshold (0.50) misses 42% of dropouts.

**Solution:** Balanced threshold of 0.35 for sustainable production use.

**Impact:**

| Threshold | Recall | Dropouts Caught | False Alarms |
|-----------|--------|-----------------|------------|
| 0.50 (default) | 58.1% | 141 / 243 | 83 |
| **0.35 (balanced)** | **72.38%** | **176 / 243** ✅ | 130 |
| 0.20 (aggressive) | 82.86% | 201 / 243 | 166 |

**Trade-off:** Balanced approach - catch most dropouts without excessive false alarms.

**Business Case:** False alarm ($500) << Dropout cost ($5000)  
**Savings:** $258,500 per 1000 patients

**Details:** [`docs/05_MODEL_TUNING.md`](docs/05_MODEL_TUNING.md)

---

## 📡 API Reference

### **Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/stats` | GET | Session stats (admin) |
| `/docs` | GET | Interactive API docs |

### **Input Schema**

Required fields (all validated):
- `patient_id` (string)
- `age` (18-85)
- `gender` (Male/Female/Non-binary)
- `treatment_group` (Active/Control/Placebo)
- `trial_phase` (Phase I/II/III)
- `days_in_trial` (> 0)
- `visits_completed` (≥ 0)
- `last_visit_day` (0 to days_in_trial)
- `adverse_events` (≥ 0)

**Full Spec:** [`docs/MODEL_IO_SPEC.md`](docs/MODEL_IO_SPEC.md)

---

## 🔧 Configuration

### **Environment Variables**

Create `.env` file (optional):
```bash
MODEL_VERSION=v3_causal
MODEL_STAGE=production
DECISION_THRESHOLD=0.35
API_PORT=8000
LOG_LEVEL=INFO
```

**No hardcoding!** All constants in [`api/config.py`](api/config.py)

**Template:** [`api/.env.example`](api/.env.example)

---

## 📊 Monitoring & Logging

### **Prediction Logs**

Every prediction logged to `logs/predictions.jsonl`:

```json
{
  "session_id": "...",
  "timestamp": "2025-12-28T22:30:00Z",
  "model_version": "v3_causal",
  "decision_threshold": 0.20,
  "patient_id": "P-1234",
  "prediction": 1,
  "probability": 0.7834,
  "risk_level": "High",
  "latency_ms": 45.2
}
```

**Purpose:** Audit trail, debugging, A/B testing, drift detection.

**Details:** [`docs/PREDICTION_LOGGING.md`](docs/PREDICTION_LOGGING.md)

---

## 📚 Documentation

**Core Guides:**
1. [`DEPLOYMENT.md`](DEPLOYMENT.md) - **Start here for deployment**
2. [`PROJECT_COMPLETE.md`](PROJECT_COMPLETE.md) - Full system overview
3. [`docs/BRANCH_STRATEGY.md`](docs/BRANCH_STRATEGY.md) - Branch organization

**Technical Docs:**
- [`docs/02_ARCHITECTURE.md`](docs/02_ARCHITECTURE.md) - System design
- [`docs/03_DATA.md`](docs/03_DATA.md) - Data engineering
- [`docs/04_ML_MODEL.md`](docs/04_ML_MODEL.md) - Model selection
- [`docs/05_MODEL_TUNING.md`](docs/05_MODEL_TUNING.md) - Threshold optimization
- [`docs/MODEL_IO_SPEC.md`](docs/MODEL_IO_SPEC.md) - API specification
- [`docs/PREDICTION_LOGGING.md`](docs/PREDICTION_LOGGING.md) - Logging architecture

---

## 🧪 Testing

### **Run Tests**
```bash
python api/test_api.py
```

Tests include:
- Health check
- High-risk patient (expect `dropout=1`)
- Low-risk patient (expect `dropout=0`)
- Session statistics

---

## 🚢 Deployment

### **Local (Development)**
```bash
python api/main.py
```

### **Docker**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]
```

### **Cloud Options**
- **AWS:** Lambda + API Gateway
- **GCP:** Cloud Run
- **Azure:** App Service

**Full Guide:** [`DEPLOYMENT.md`](DEPLOYMENT.md)

---

## 🏢 Enterprise Features

### ✅ **Production-Ready**
- FastAPI with input validation
- MLflow model registry (no `.pkl` files)
- DVC data versioning
- Server-side audit logging
- Configuration via environment variables

### ✅ **MLOps Best Practices**
- Experiment tracking (MLflow)
- Model versioning
- Data lineage
- Cost-sensitive optimization
- Clean branch separation (main/research)

### ✅ **Industry Alignment**
Matches practices at:
- Google (clean prod + research forks)
- Meta (service isolation)
- Pharmaceutical ML (compliance)
- FinTech (security standards)

---

## 📈 Key Learnings

1. **Model Selection:** LightGBM > XGBoost > Logistic Regression for recall-focused tasks
2. **Threshold Tuning:** Balanced threshold of 0.35 provides good recall (72%) with acceptable precision
3. **Feature Engineering:** Rate-based features > raw counts for causal data
4. **Branch Strategy:** Strict production/research separation keeps deployment clean
5. **Configuration:** Zero hardcoding enables easy A/B testing and deployment

---

## 🔄 Workflow

### **For Development**
```bash
git checkout research          # Experiments branch
python src/experiments/threshold_tuning.py
```

### **For Deployment**
```bash
git checkout main              # Production branch
python api/main.py
```

Knowledge documented in `main`, experiments in `research`.

---

## 📋 Requirements

**Python:** 3.10+

**Key Dependencies:**
- FastAPI + Uvicorn (API)
- LightGBM + XGBoost (models)
- MLflow (experiment tracking)
- DVC (data versioning)
- Pydantic (validation)

**Full list:** [`requirements.txt`](requirements.txt)

---

## 🎯 Future Enhancements

**Planned (in research branch):**
- [ ] Confidence intervals on predictions
- [ ] Dynamic threshold adjustment
- [ ] Real-time monitoring dashboard
- [ ] A/B testing framework
- [ ] Multi-tier risk stratification

---

## 📞 Support

**GitHub:** [MLOps-System-for-Clinical-Trial-Risk-Prediction](https://github.com/Jayasudhandesigner/MLOps-System-for-Clinical-Trial-Risk-Prediction)

**Branches:**
- `main` - Production-ready deployment
- `research` - Experiments & analysis

**Issues:** Check [`DEPLOYMENT.md`](DEPLOYMENT.md) troubleshooting section

---

## ✨ Achievement Summary

**What This System Delivers:**

✅ **Production-ready API** with FastAPI + server-side logging  
✅ **82.86% recall** catching 83% of dropout cases early  
✅ **$258K savings** per 1000 patients in intervention costs  
✅ **Enterprise-grade** branch separation and MLOps practices  
✅ **Zero hardcoding** - fully configurable via environment  
✅ **Complete documentation** - deployment guide + knowledge base  

---

## 📄 License

MIT License

---

## 📊 Status

**Version:** 1.0  
**Status:** ✅ **Production-Ready**  
**Last Updated:** 2025-12-28  
**Branches:** `main` (deployment), `research` (experiments)

🎉 **Ready for deployment and interviews!**
