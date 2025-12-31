# Day 15: Monitoring, Drift & Decay Detection - Implementation Summary

## 🎯 What Was Achieved

Day 15 implements **Post-Deployment Monitoring** with:

1. ✅ Data drift detection
2. ✅ Prediction drift detection
3. ✅ Alert triggering system
4. ✅ Retraining policy documentation
5. ✅ CI/CD integration-ready scripts

**Philosophy**: "Models rot. Systems must notice."

---

## 📁 Files Created

```
monitoring/
 ├── __init__.py                 # Package init
 ├── build_reference.py          # Build reference dataset
 ├── prediction_monitor.py       # Live prediction logger
 ├── run_drift.py                # Drift detection report generator
 └── alert_trigger.py            # CI/CD-ready alert script

docs/
 └── monitoring_policy.md        # Governance & retraining policy
```

---

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
pip install evidently pandas scikit-learn
pip freeze > requirements.lock.txt
```

### 2. Build Reference Dataset

```bash
python monitoring/build_reference.py
```

This creates `monitoring/reference.csv` from training data.

### 3. Version Reference with DVC

```bash
dvc add monitoring/reference.csv
git add monitoring/reference.csv.dvc .gitignore
git commit -m "monitoring: add reference dataset for drift detection"
```

### 4. Start Making Predictions

The API automatically logs predictions to `monitoring/live_predictions.jsonl`.

### 5. Run Drift Detection

```bash
python monitoring/run_drift.py
```

Outputs:
- `monitoring/drift_report.html` (visual report)
- `monitoring/drift_report.json` (for CI/CD)

---

## 🚨 Alert Thresholds

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Feature drift share | >30% | Investigate |
| Critical feature drift | Any of `age`, `adverse_events`, `days_in_trial` | Alert |
| Prediction shift | >20% | Investigate |
| Recall drop | <0.55 | **Retrain immediately** |

---

## 📊 What Gets Monitored

| Type | Question | Method |
|------|----------|--------|
| **Data Drift** | Has input data changed? | Compare feature distributions |
| **Prediction Drift** | Is model behaving differently? | Compare prediction distributions |
| **Performance Drift** | Is model getting worse? | Compare metrics to baseline |

---

## 🔄 Workflow

```
1. API receives prediction request
       ↓
2. Model makes prediction
       ↓
3. Prediction logged to JSONL ← NEW (Day 15)
       ↓
4. Response returned to user
       ↓
5. Periodic drift analysis ← NEW (Day 15)
       ↓
6. Alert if drift detected ← NEW (Day 15)
       ↓
7. Retrain if needed (through CI/CD)
```

---

## 🏃 Running Locally

```bash
# Build reference (once after training)
python monitoring/build_reference.py

# Make some predictions (via API)
curl -X POST http://localhost:8000/predict ...

# Run drift detection
python monitoring/run_drift.py

# Check alerts (for CI/CD)
python monitoring/alert_trigger.py
```

---

## 🔗 CI/CD Integration

Add to your workflow:

```yaml
- name: Check for drift
  run: |
    python monitoring/alert_trigger.py
```

Exit codes:
- `0` = No drift (continue)
- `1` = Drift detected (investigate/retrain)

---

## 🏥 Healthcare Compliance

This monitoring system supports:

- ✅ **Audit Trail**: All predictions logged with timestamps
- ✅ **Drift Detection**: Distribution shift monitoring
- ✅ **Performance Tracking**: Metric degradation alerts
- ✅ **Documentation**: Retraining policy defined
- ✅ **Human Override**: Manual intervention always possible

---

## ⚠️ Important Notes

1. Reference dataset should be updated after each retraining
2. Drift ≠ failure; drift = decision point
3. Always investigate root cause before retraining
4. New models must pass CI/CD quality gates

---

**Day 15 Complete!** 🎉

Your system now:
- Monitors for model degradation
- Detects data and prediction drift
- Triggers alerts when needed
- Has documented retraining policies
- Is compliant with healthcare ML requirements
