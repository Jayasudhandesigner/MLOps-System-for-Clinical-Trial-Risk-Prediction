# Prediction Logging & Traceability

**Server-Side Audit Trail for Model Governance**

---

## Overview

Every prediction made by the API is logged with **full metadata** for traceability, debugging, and compliance. This logging is **server-side only** - users never see model internals.

---

## What Gets Logged

### Complete Log Entry Example

```json
{
  "session_id": "a7f3c8d2-1e4b-4a9c-8f2d-9c3e1b5a7f8e",
  "timestamp": "2025-12-28T22:09:35Z",
  "model_version": "v3_causal",
  "model_stage": "production",
  "decision_threshold": 0.20,
  "patient_id": "P-1234",
  "input_hash": "7a3f8c2d1e4b9a8f",
  "prediction": 1,
  "probability": 0.783456,
  "risk_level": "High",
  "latency_ms": 45.23
}
```

### Metadata Fields

| Field | Type | Purpose | Exposed to User? |
|-------|------|---------|------------------|
| `session_id` | UUID | Track API server session | ❌ No |
| `timestamp` | ISO8601 | When prediction was made | ❌ No |
| `model_version` | string | Feature version used | ❌ No |
| `model_stage` | string | production/staging | ❌ No |
| `decision_threshold` | float | Binary decision cutoff | ❌ No |
| `patient_id` | string | Patient identifier | ✅ Yes |
| `input_hash` | string | SHA256 hash of input | ❌ No |
| `prediction` | int | Binary prediction (0/1) | ✅ Yes (as dropout_prediction) |
| `probability` | float | Dropout probability | ❌ No (only risk_level shown) |
| `risk_level` | string | Low/Moderate/High/Critical | ✅ Yes |
| `latency_ms` | float | Prediction time | ❌ No |

---

## User-Facing vs Internal Logging

### ✅ What Users See (API Response)

```json
{
  "patient_id": "P-1234",
  "dropout_prediction": 1,
  "risk_level": "High",
  "recommended_action": "weekly_monitoring"
}
```

**No exposure of:**
- ❌ Model version
- ❌ Decision threshold
- ❌ Model stage
- ❌ Exact probability
- ❌ Session ID
- ❌ Latency

### 🔒 What Gets Logged (Server-Side Only)

**Full audit trail in `logs/predictions.jsonl`:**

```json
{"session_id": "...", "timestamp": "...", "model_version": "v3_causal", "decision_threshold": 0.20, ...}
{"session_id": "...", "timestamp": "...", "model_version": "v3_causal", "decision_threshold": 0.20, ...}
{"session_id": "...", "timestamp": "...", "model_version": "v3_causal", "decision_threshold": 0.20, ...}
```

**Format:** JSON Lines (one JSON object per line)  
**Location:** `logs/predictions.jsonl`  
**Access:** Server administrators only

---

## Why This Matters

### 1. **Model Governance**

**Auditing:**
- Track which model version made which prediction
- Verify correct threshold was used
- Prove compliance with clinical protocols

**Example Use Case:**
> "A patient complained about a retention call. Which model version flagged them? Was threshold=0.20 or 0.25 used?"

**Query:**
```bash
grep "P-1234" logs/predictions.jsonl | jq '{patient_id, model_version, decision_threshold, prediction}'
```

---

### 2. **A/B Testing**

**Scenario:** Testing threshold 0.20 vs 0.25 on different cohorts

**Server creates two sessions:**
- Session A: threshold=0.20
- Session B: threshold=0.25

**Users see identical responses**, but logs track which threshold was used:

```python
# Session comparison
session_a_stats = get_session_stats("session-A-uuid")
session_b_stats = get_session_stats("session-B-uuid")

# Compare positive prediction rates
print(f"Threshold 0.20: {session_a_stats['positive_rate']}")
print(f"Threshold 0.25: {session_b_stats['positive_rate']}")
```

---

### 3. **Debugging & Troubleshooting**

**Problem:** "Model seems to be overpredicting dropouts"

**Investigation:**
```python
import json

# Load recent predictions
with open('logs/predictions.jsonl') as f:
    predictions = [json.loads(line) for line in f]

# Check average probability
avg_prob = sum(p['probability'] for p in predictions) / len(predictions)
print(f"Average dropout probability: {avg_prob:.3f}")

# Check threshold usage
thresholds = {p['decision_threshold'] for p in predictions}
print(f"Thresholds in use: {thresholds}")
```

---

### 4. **Performance Monitoring**

**Track latency over time:**

```python
import pandas as pd

# Load logs
df = pd.read_json('logs/predictions.jsonl', lines=True)

# Latency statistics
print(df['latency_ms'].describe())

# Alert if p95 latency > 100ms
p95_latency = df['latency_ms'].quantile(0.95)
if p95_latency > 100:
    alert("High latency detected")
```

---

### 5. **Model Drift Detection**

**Compare prediction distribution over time:**

```python
# Week 1 predictions
week1 = df[df['timestamp'].between('2025-01-01', '2025-01-07')]

# Week 2 predictions
week2 = df[df['timestamp'].between('2025-01-08', '2025-01-14')]

# Compare positive prediction rates
drift = abs(week2['prediction'].mean() - week1['prediction'].mean())
if drift > 0.05:
    alert(f"Model drift detected: {drift:.1%} change in positive rate")
```

---

## Implementation in API

### Initialization (on API startup)

```python
from api.prediction_logger import init_logger
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Initialize logger with model metadata
    init_logger(
        model_version="v3_causal",
        model_stage="production",
        decision_threshold=0.20,
        log_file="logs/predictions.jsonl"
    )
    print("✅ Prediction logger initialized")
```

### Prediction Endpoint

```python
from api.prediction_logger import get_logger
import time

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    # 1. Feature engineering
    features = engineer_features(request.dict())
    
    # 2. Prediction
    probability = model.predict_proba(features)[0, 1]
    prediction = int(probability >= 0.20)
    risk_level = get_risk_level(probability)
    
    # 3. SERVER-SIDE LOGGING (invisible to user)
    logger = get_logger()
    logger.log_prediction(
        patient_id=request.patient_id,
        input_data=request.dict(),
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
        latency_ms=(time.time() - start_time) * 1000
    )
    
    # 4. USER-FACING RESPONSE (clean, no metadata)
    return {
        "patient_id": request.patient_id,
        "dropout_prediction": prediction,
        "risk_level": risk_level,
        "recommended_action": get_action(probability)
    }
```

**Key Points:**
- ✅ Logging happens **after** prediction (doesn't slow down response)
- ✅ User never sees threshold, model_version, or probability
- ✅ Full audit trail preserved server-side

---

## Log File Management

### File Format

**JSON Lines (`.jsonl`):**
- One JSON object per line
- Easy to stream/append
- Standard format for log aggregation tools

### Rotation Strategy

**Recommended:**
```python
# Daily rotation
# logs/predictions.2025-12-28.jsonl
# logs/predictions.2025-12-29.jsonl
```

**Python implementation:**
```python
from datetime import datetime

log_file = f"logs/predictions.{datetime.now():%Y-%m-%d}.jsonl"
```

### Retention Policy

| Stage | Retention | Purpose |
|-------|-----------|---------|
| Production | 90 days | Compliance, debugging |
| Archive | 1 year | Audit trail |
| Deleted | After 1 year | Comply with data retention policies |

---

## Analysis Examples

### 1. Daily Prediction Report

```python
import pandas as pd

# Load today's predictions
df = pd.read_json('logs/predictions.2025-12-28.jsonl', lines=True)

# Statistics
report = {
    "total_predictions": len(df),
    "positive_predictions": df['prediction'].sum(),
    "positive_rate": df['prediction'].mean(),
    "avg_probability": df['probability'].mean(),
    "avg_latency_ms": df['latency_ms'].mean(),
    "risk_distribution": df['risk_level'].value_counts().to_dict()
}

print(report)
```

### 2. Session Comparison

```python
# Get stats for specific session
session_stats = logger.get_session_stats()

print(f"""
Session: {session_stats['session_id']}
Model: {session_stats['model_version']}
Threshold: {session_stats['threshold']}
Predictions: {session_stats['total_predictions']}
Positive Rate: {session_stats['positive_rate']:.1%}
Avg Latency: {session_stats['avg_latency_ms']:.1f}ms
""")
```

### 3. Error Tracking

```python
# Log errors
logger.log_error(
    patient_id="P-ERROR",
    error_type="ValueError",
    error_message="Invalid age: -5",
    input_data={"age": -5, ...}
)

# Query errors
errors = [json.loads(line) for line in open('logs/predictions.jsonl') 
          if 'error_type' in json.loads(line)]
print(f"Total errors: {len(errors)}")
```

---

## Security Considerations

### ✅ Best Practices

1. **No PII in Logs** (if required by regulation)
   - Hash patient IDs: `patient_id: sha256(patient_id)[:16]`
   - Don't log raw input data, only hash

2. **Access Control**
   - Logs accessible only to authorized personnel
   - File permissions: `chmod 600 logs/predictions.jsonl`

3. **Encryption**
   - Encrypt logs at rest
   - Use secure log aggregation (e.g., CloudWatch with encryption)

4. **API Security**
   - Never expose `/logs` endpoint publicly
   - Authentication required for admin endpoints

---

## Integration with Monitoring Tools

### Export to Prometheus

```python
from prometheus_client import Counter, Histogram

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
latency_histogram = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
async def predict(request: PredictionRequest):
    with latency_histogram.time():
        # ... prediction logic ...
        prediction_counter.inc()
```

### Export to ElasticSearch

```python
from elasticsearch import Elasticsearch

es = Elasticsearch()

def log_to_es(log_entry):
    es.index(
        index="predictions",
        document=log_entry
    )
```

---

## Summary

**What This System Provides:**

✅ **Full traceability** - Every prediction logged with metadata  
✅ **User privacy** - Internal details hidden from API responses  
✅ **Debugging** - Investigate issues with complete context  
✅ **Governance** - Audit trail for compliance  
✅ **A/B testing** - Track different thresholds/models  
✅ **Performance monitoring** - Latency and drift detection  

**Critical Principle:**
> "Log everything server-side, show only what's necessary to users."

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-28  
**Log Format:** JSON Lines  
**Default Location:** `logs/predictions.jsonl`
