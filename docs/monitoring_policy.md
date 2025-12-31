# Drift & Retraining Policy

## Overview

This document defines the monitoring and retraining policy for the Clinical Trial Dropout Prediction system.

**Philosophy**: "Models rot. Systems must notice."

---

## 1. What We Monitor

| Type | Question | How |
|------|----------|-----|
| **Data Drift** | Has input data changed? | Compare feature distributions |
| **Prediction Drift** | Is the model behaving differently? | Compare prediction distributions |
| **Performance Drift** | Is the model getting worse? | Compare metrics against baseline |

Most teams only monitor data drift. **We monitor all three.**

---

## 2. Alert Thresholds

### Data Drift Alerts

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Feature drift share | >30% of features | Investigate |
| Critical feature drift | `age`, `adverse_events`, `days_in_trial` | Alert immediately |
| Prediction distribution shift | >20% | Investigate |

### Performance Alerts

| Metric | Minimum | Action if Below |
|--------|---------|-----------------|
| Recall | 0.55 | **Retrain immediately** |
| ROC-AUC | 0.60 | Investigate + potential retrain |
| Precision | 0.30 | Investigate |

---

## 3. Retraining Triggers

Retraining is **required** when:

1. ✅ Recall drops below 0.55
2. ✅ >30% features show significant drift
3. ✅ Any critical feature (`age`, `adverse_events`) drifts
4. ✅ Prediction distribution shifts >20%

Retraining is **recommended** when:

1. ⚠️ ROC-AUC drops below 0.60
2. ⚠️ 15-30% features show drift
3. ⚠️ Monthly scheduled review indicates degradation

---

## 4. Retraining Rules

### Pre-retraining Checklist

- [ ] Confirm drift is real (not data quality issue)
- [ ] Document root cause if known
- [ ] Verify new training data is available
- [ ] Ensure CI/CD pipeline is ready

### Retraining Process

1. Trigger retraining pipeline
2. New model must pass ALL CI/CD quality gates
3. Compare new model vs current production model
4. **Only promote if new model is better**
5. Document the change

### Post-retraining

- [ ] Log retraining event in MLflow
- [ ] Update reference dataset
- [ ] Notify stakeholders
- [ ] Monitor new model closely for 48 hours

---

## 5. Monitoring Schedule

| Activity | Frequency | Owner |
|----------|-----------|-------|
| Drift report generation | Daily | Automated |
| Alert review | Daily | ML Engineer |
| Full drift analysis | Weekly | Data Science Team |
| Model performance review | Weekly | ML Engineer |
| Reference dataset update | After each retrain | ML Engineer |

---

## 6. Escalation Path

| Severity | Response Time | Escalation |
|----------|---------------|------------|
| Info | 24 hours | Log only |
| Warning | 4 hours | ML Engineer |
| Critical | 1 hour | ML Lead + Stakeholders |

---

## 7. Compliance Requirements

For healthcare ML systems:

1. **Audit Trail**: All predictions are logged with timestamps
2. **Explainability**: Model decisions can be traced to features
3. **Drift Detection**: System monitors for distribution shifts
4. **Human Override**: Manual intervention always possible
5. **Documentation**: All retraining events are documented

---

## 8. Tools Used

| Tool | Purpose |
|------|---------|
| Evidently AI | Drift detection and reporting |
| MLflow | Experiment tracking and model registry |
| GitHub Actions | CI/CD pipeline |
| JSONL Logs | Prediction audit trail |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-31 | Initial policy (Day 15) |

---

*This policy is part of the MLOps System for Clinical Trial Risk Prediction.*
