# Model Input/Output Specification

**LightGBM Clinical Trial Dropout Prediction Model**

---

## Model Overview

**Model Type:** LightGBM Binary Classifier  
**Task:** Predict patient dropout risk in clinical trials  
**Decision Threshold:** 0.20 (optimized for recall)  
**Version:** v3_causal  

---

## Input Specification

### Required Input Features (9 features)

The model expects **9 numerical features** after preprocessing. These are derived from raw patient data.

#### 1. **Base Features** (2)

| Feature | Type | Range | Description | Example |
|---------|------|-------|-------------|---------|
| `age` | int | 18-85 | Patient age in years | 65 |
| `days_in_trial` | int | 30-365 | Days since enrollment | 120 |

#### 2. **Rate Features** (3)

| Feature | Type | Range | Description | Calculation | Example |
|---------|------|-------|-------------|-------------|---------|
| `visit_rate` | float | 0.0-1.5 | Compliance ratio | `visits_completed / (days_in_trial/30)` | 0.83 |
| `adverse_event_rate` | float | 0.0-0.1 | Event frequency | `adverse_events / days_in_trial` | 0.025 |
| `time_since_last_visit` | int | 0-365 | Days since last visit | `days_in_trial - last_visit_day` | 15 |

#### 3. **Interaction Features** (2)

| Feature | Type | Range | Description | Calculation | Example |
|---------|------|-------|-------------|-------------|---------|
| `burden` | float | 0.0-0.1 | Combined stress | `adverse_event_rate × (1 - visit_rate)` | 0.004 |
|`age_adverse_risk` | float | 0.0-0.1 | Age-weighted risk | `(age/85) × adverse_event_rate` | 0.019 |

#### 4. **Domain Risk Features** (2)

| Feature | Type | Range | Description | Mapping | Example |
|---------|------|-------|-------------|---------|---------|
| `trial_phase_risk` | float | 0.2-0.8 | Phase difficulty | Phase I: 0.2, Phase II: 0.5, Phase III: 0.8 | 0.8 |
| `treatment_risk` | float | 0.1-0.4 | Group frustration | Active: 0.1, Control: 0.3, Placebo: 0.4 | 0.4 |

---

## Raw Input Format (for API)

### Example API Request Payload

```json
{
  "patient_id": "P-1234",
  "age": 65,
  "gender": "Female",
  "treatment_group": "Placebo",
  "trial_phase": "Phase III",
  "days_in_trial": 120,
  "visits_completed": 3,
  "last_visit_day": 105,
  "adverse_events": 4
}
```

### Required Raw Fields

| Field | Type | Required | Valid Values | Description |
|-------|------|----------|--------------|-------------|
| `patient_id` | string | Yes | Any | Unique patient identifier |
| `age` | int | Yes | 18-85 | Patient age |
| `gender` | string | No | "Male", "Female", "Non-binary" | Patient gender (not used by model) |
| `treatment_group` | string | Yes | "Active", "Control", "Placebo" | Treatment assignment |
| `trial_phase` | string | Yes | "Phase I", "Phase II", "Phase III" | Current trial phase |
| `days_in_trial` | int | Yes | \u003e 0 | Days since enrollment |
| `visits_completed` | int | Yes | ≥ 0 | Total completed visits |
| `last_visit_day` | int | Yes | 0 to days_in_trial | Day of last visit (0 if no visits) |
| `adverse_events` | int | Yes | ≥ 0 | Total adverse events reported |

---

## Feature Engineering Pipeline

**Transformation:** Raw Input → Preprocessed Features → Model Input

```python
# 1. Rate Features
visit_rate = visits_completed / (days_in_trial / 30)
adverse_event_rate = adverse_events / days_in_trial
time_since_last_visit = days_in_trial - last_visit_day

# 2. Interaction Features
burden = adverse_event_rate * (1 - visit_rate)
age_adverse_risk = (age / 85) * adverse_event_rate

# 3. Domain Features
trial_phase_risk = {"Phase I": 0.2, "Phase II": 0.5, "Phase III": 0.8}[trial_phase]
treatment_risk = {"Active": 0.1, "Control": 0.3, "Placebo": 0.4}[treatment_group]

# 4. Standardization (StandardScaler)
# All 9 features are scaled: (x - mean) / std
```

---

## Output Specification

### Model Outputs

The model returns **2 values**:

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `dropout_probability` | float | 0.0-1.0 | Probability of dropout (class 1) |
| `dropout_prediction` | int | 0 or 1 | Binary prediction (using threshold 0.20) |

### Prediction Logic

```python
dropout_probability = model.predict_proba(X)[:, 1]  # P(dropout=1)
dropout_prediction = (dropout_probability >= 0.20).astype(int)
```

**Decision Rule:**
- `dropout_probability >= 0.20` → **Predict Dropout (1)**
- `dropout_probability < 0.20` → **Predict Retention (0)**

---

## Example API Response

### Success Response

```json
{
  "patient_id": "P-1234",
  "dropout_probability": 0.7834,
  "dropout_prediction": 1,
  "risk_level": "High",
  "threshold_used": 0.20,
  "recommended_action": "immediate_intervention",
  "model_version": "v3_causal",
  "timestamp": "2025-12-28T21:58:00Z"
}
```

### Risk Stratification

| Probability Range | Risk Level | Prediction | Action |
|-------------------|------------|------------|--------|
| P ≥ 0.40 | **Critical** | 1 | Immediate personal intervention |
| 0.20 ≤ P < 0.40 | **High** | 1 | Weekly monitoring + automated reminders |
| 0.10 ≤ P < 0.20 | **Moderate** | 0 | Biweekly check-in |
| P < 0.10 | **Low** | 0 | Standard protocol |

---

## Data Validation Rules

### Input Validation

**Age:**
- Must be integer
- Range: 18-85
- Invalid: Raise 400 error

**Days in Trial:**
- Must be positive integer
- Range: \u003e 0
- Invalid: Raise 400 error

**Visits Completed:**
- Must be non-negative integer
- Range: ≥ 0
- Warning if 0 (no engagement)

**Adverse Events:**
- Must be non-negative integer
- Range: ≥ 0
- Warning if \u003e 10 (unusual)

**Categorical Fields:**
- Must match exact strings (case-sensitive)
- Invalid: Raise 400 error with valid options

### Derived Feature Validation

**Visit Rate:**
- Expected range: 0.0-1.5
- \u003e 2.0: Log warning (data quality issue)

**Adverse Event Rate:**
- Expected range: 0.0-0.1
- \u003e 0.15: Log warning (high event frequency)

---

## Example Use Cases

### 1. **High-Risk Patient (Likely Dropout)**

**Input:**
```json
{
  "age": 72,
  "treatment_group": "Placebo",
  "trial_phase": "Phase III",
  "days_in_trial": 150,
  "visits_completed": 2,
  "last_visit_day": 90,
  "adverse_events": 8
}
```

**Derived Features:**
- visit_rate = 2 / (150/30) = 0.40 (poor compliance)
- adverse_event_rate = 8 / 150 = 0.053 (high events)
- burden = 0.053 × (1 - 0.40) = 0.032 (high)
- trial_phase_risk = 0.8 (Phase III)
- treatment_risk = 0.4 (Placebo)

**Output:**
```json
{
  "dropout_probability": 0.89,
  "dropout_prediction": 1,
  "risk_level": "Critical"
}
```

---

### 2. **Low-Risk Patient (Likely Retention)**

**Input:**
```json
{
  "age": 45,
  "treatment_group": "Active",
  "trial_phase": "Phase I",
  "days_in_trial": 60,
  "visits_completed": 2,
  "last_visit_day": 55,
  "adverse_events": 1
}
```

**Derived Features:**
- visit_rate = 2 / (60/30) = 1.0 (good compliance)
- adverse_event_rate = 1 / 60 = 0.017 (low events)
- burden = 0.017 × (1 - 1.0) = 0.0 (minimal)
- trial_phase_risk = 0.2 (Phase I)
- treatment_risk = 0.1 (Active)

**Output:**
```json
{
  "dropout_probability": 0.08,
  "dropout_prediction": 0,
  "risk_level": "Low"
}
```

---

## Preprocessing Artifacts

### Saved Preprocessor

**File:** `data/processed/preprocessor_dropout_v3_causal.pkl`

**Type:** sklearn ColumnTransformer with StandardScaler

**Usage:**
```python
import joblib
preprocessor = joblib.load('data/processed/preprocessor_dropout_v3_causal.pkl')
X_scaled = preprocessor.transform(X_engineered)
```

**Note:** The preprocessor expects **9 engineered features** (NOT raw input). Feature engineering must be done before applying the preprocessor.

---

## Model Artifacts

### Saved Model

**MLflow Model Registry:** `ClinicalTrialDropout_dropout_v3_causal`  
**Latest Version:** v6  
**Run ID:** Available in MLflow UI  

**Load Model:**
```python
import mlflow
model = mlflow.sklearn.load_model(f"models:/ClinicalTrialDropout_dropout_v3_causal/6")
```

---

## Performance Metrics (Threshold 0.20)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Recall** | 82.86% | Catches 83% of all dropouts |
| **Precision** | 54.72% | 55% of flagged patients drop out |
| **F1 Score** | 0.6591 | Balanced performance |
| **ROC-AUC** | 0.6182 | Good separation |

**Clinical Impact:**
- On 1000 patients with 243 dropouts:
  - **201 dropouts caught** (83%)
  - **42 dropouts missed** (17%)
  - **166 false alarms** (acceptable for intervention)

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-28  
**Model Version:** v3_causal  
**Decision Threshold:** 0.20
