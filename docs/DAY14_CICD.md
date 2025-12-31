# Day 14: CI/CD for ML - Implementation Summary

## 🎯 What Was Achieved

Day 14 implements **Guarded Automation** - a production-grade CI/CD pipeline that:

1. ✅ Runs tests on every PR and push to main
2. ✅ Validates data + model behavior automatically
3. ✅ Trains model with MLflow tracking
4. ✅ Enforces quality gates before promotion
5. ✅ Never deploys a worse model

## 📁 Files Created

```
.github/
 └── workflows/
     └── mlops-ci.yml          # GitHub Actions workflow (6 stages)

tests/
 ├── __init__.py               # Package init
 ├── test_data.py              # Data quality validation (8 tests)
 ├── test_features.py          # Feature engineering validation (10 tests)
 └── test_model.py             # Model performance validation (5 tests)

scripts/
 ├── __init__.py               # Package init
 └── validate_metrics.py       # CRITICAL: Quality gate script
```

## 🔒 Quality Gates

The system enforces these minimum thresholds:

| Metric    | Minimum | Purpose |
|-----------|---------|---------|
| Recall    | 0.60    | Catch dropout cases |
| ROC-AUC   | 0.60    | Overall discrimination |
| Precision | 0.30    | Avoid false positives |
| F1 Score  | 0.40    | Balance |

## 🧪 Test Categories

### Data Tests (`test_data.py`)
- Dataset size validation (>100 samples)
- Target distribution (5-50% dropout rate)
- Patient ID integrity
- Binary target validation
- Required columns present
- No duplicate IDs
- Age range validation
- Days in trial validation

### Feature Tests (`test_features.py`)
- Non-negative counts (visits, adverse events)
- Date consistency (last_visit_day vs days_in_trial)
- Categorical validation (phase, treatment, gender)
- No NaN in numeric features
- Dropout day logic
- Feature correlations with target

### Model Tests (`test_model.py`)
- Recall exceeds baseline
- ROC-AUC exceeds threshold
- Precision acceptable
- Model artifacts exist
- CV variance reasonable

## 🚀 CI/CD Pipeline Stages

```
1. Data Validation      → Ensures data quality
2. Feature Validation   → Ensures feature correctness
3. Model Training       → Trains with MLflow tracking
4. Quality Gate Check   → Blocks bad models
5. Integration Tests    → Tests API endpoints
6. Docker Build         → Creates container (main only)
```

## 🏃 Running Locally

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_data.py -v
pytest tests/test_features.py -v
pytest tests/test_model.py -v

# Validate model metrics
python scripts/validate_metrics.py
```

## 🔄 What Happens on PR/Push

1. **PR to main**: Full pipeline runs, Docker build skipped
2. **Push to main**: Full pipeline + Docker build
3. **Any failure**: Pipeline stops, PR blocked

## ⚠️ Important Notes

- Model tests skip if no MLflow runs exist
- DVC pull is optional (fallback to synthetic data generation)
- Quality gates are configurable in `validate_metrics.py`
- Docker push is commented out (configure registry secrets first)

## 📊 Philosophy

> "Automation over trust."

We are doing:
- ✅ Guarded automation
- ✅ Metric-based promotion
- ✅ Human override possible

We are NOT doing:
- ❌ Auto-deploy on every commit
- ❌ Blind retraining

---

**Day 14 Complete!** 🎉

You now have a production-grade ML system with:
- Automatic regression prevention
- Data sanity enforcement
- Metric-based approval
- Safe retraining
- CI that understands **ML**, not just code
