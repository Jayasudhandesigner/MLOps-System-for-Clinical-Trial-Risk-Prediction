# Data Specification

**Clinical Trial Patient Dropout Dataset**

---

## Dataset Overview

**Source:** Synthetic generation with causal relationships  
**Size:** 1000 patient records  
**Purpose:** Train ML models to predict dropout risk  
**Generation:** `data/synthetic_data_causal.py`

---

## Schema

### Raw Data Columns (14 fields)

| Column | Type | Range/Values | Description |
|--------|------|--------------|-------------|
| `patient_id` | string | P-0001 to P-1000 | Unique patient identifier |
| `age` | int | 18-85 | Patient age in years |
| `gender` | categorical | Male, Female, Non-binary | Patient gender |
| `treatment_group` | categorical | Active, Control, Placebo | Assigned treatment |
| `trial_phase` | categorical | Phase I, Phase II, Phase III | Trial phase |
| `days_in_trial` | int | 30-365 | Days since enrollment |
| `visits_completed` | int | 0-12 | Number of clinic visits |
| `last_visit_day` | int | 0-365 | Day of most recent visit |
| `adverse_events` | int | 0-10 | Number of side effects |
| `dropout` | binary | 0, 1 | Target: Did patient drop out? |
| `dropout_day` | int | 0-365 | Day dropout occurred (0 if no dropout) |
| `early_dropout` | binary | 0, 1 | Dropout < 90 days |
| `late_dropout` | binary | 0, 1 | Dropout ≥ 90 days |
| `dropout_30_days` | binary | 0, 1 | Dropout ≤ 30 days |

---

## Causal Data Generation

### Risk Score Calculation

Dropout probability determined by weighted risk factors:

```python
risk_score = (
    0.35 * adverse_event_rate +        # Normalized side effects
    0.30 * (1 - visit_rate) +           # Poor compliance
    0.20 * phase_risk +                 # Trial duration stress
    0.10 * treatment_risk +             # Placebo frustration
    0.05 * youth_factor                 # Young patient dropout
)

# Convert to probability (sigmoid)
dropout_probability = 1 / (1 + exp(-3 * (risk_score - 0.5)))

# Assign dropout
dropout = 1 if random() < dropout_probability else 0
```

### Causal Relationships

**Adverse Events → Dropout:**
- High adverse_event_rate (> 0.1/day) → +30% dropout risk
- Older patients (age > 65) + adverse events → compound risk

**Visit Compliance → Dropout:**
- Low visit_rate (< 0.5) → +20% dropout risk
- Missed visits signal disengagement

**Trial Phase → Dropout:**
- Phase I (short): 20% base risk
- Phase II (medium): 50% base risk
- Phase III (long): 80% base risk

**Treatment Group → Dropout:**
- Active (benefits seen): 10% base risk
- Control (mixed): 30% base risk
- Placebo (no benefits): 40% base risk

---

## Engineered Features (7 additional)

### Rate Features (Temporal Normalization)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `visit_rate` | visits / (days/30 + 1) | Compliance vs expected |
| `adverse_event_rate` | events / (days + 1) | Risk normalized by time |
| `time_since_last_visit` | days - last_visit_day | Engagement gap |

### Interaction Features (Compound Effects)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `burden` | adverse_rate × (1 - visit_rate) | Patient under stress |
| `age_adverse_risk` | (age/85) × adverse_rate | Age-dependent tolerance |

### Domain Encoding (Ordinal Risk)

| Feature | Mapping | Purpose |
|---------|---------|---------|
| `trial_phase_risk` | {I: 0.2, II: 0.5, III: 0.8} | Duration burden |
| `treatment_risk` | {Active: 0.1, Control: 0.3, Placebo: 0.4} | Benefit perception |

---

## Data Quality Metrics

### Distribution Statistics

```
Dropout Rate:         24.3%
Early Dropout:        15.1%
Late Dropout:         9.2%
30-Day Dropout:       6.4%

Age:                  Mean 51.2, Std 19.4
Days in Trial:        Mean 197.3, Std 105.6
Visits Completed:     Mean 6.5, Std 3.8
Adverse Events:       Mean 1.4, Std 1.2
```

### Feature Correlations with Dropout

```
Feature                 Correlation
visit_rate             -0.342  (negative: good compliance → less dropout)
adverse_event_rate      0.287  (positive: more events → more dropout)
burden                  0.310  (positive: high burden → more dropout)
trial_phase_risk        0.185  (positive: longer trial → more dropout)
age                     0.042  (weak: age not primary factor)
```

**Learnable Signal:** Maximum absolute correlation 0.342 (>> 0.15 threshold)

---

## Data Validation Rules

### Ingest Validation (`src/core/ingest.py`)

1. **Schema Check:** All 14 columns present
2. **Patient ID:** Unique, no nulls
3. **Dropout:** Binary values only (0 or 1)
4. **Data Types:** Correct types for each column
5. **Range Checks:** Values within specified ranges

### Preprocessing Validation

1. **Missing Values:** Imputed with median (numeric) or mode (categorical)
2. **Outliers:** Detected but retained (clinical data can have extremes)
3. **Scaling:** StandardScaler applied to numeric features

---

## Reproducibility

### Deterministic Generation

```python
np.random.seed(42)  # Fixed seed
```

**Result:** Same data on every run

### Verification

Check correlation after generation:
```bash
python data/synthetic_data_causal.py

Expected output:
✅ Maximum absolute correlation: 0.34 (> 0.15)
✅ Learnable signal exists
```

---

## Data Splits

### Train/Test Split

```
Total:     1000 patients
Train:     800 patients (80%)
Test:      200 patients (20%)
Strategy:  Stratified (maintains 24.3% dropout in both)
```

### Cross-Validation

```
Method:    StratifiedKFold
Folds:     5
Purpose:   Robust performance estimation
```

---

## Future Enhancements

### Real Data Integration

When using actual clinical trial data:

1. **Privacy:** Remove PII, use patient_id hashing
2. **Compliance:** HIPAA-compliant data handling
3. **Augmentation:** Combine with synthetic for training stability
4. **Drift Monitoring:** Evidently AI for distribution shifts

### Additional Features

Potential features from real trials:

- Medication adherence scores
- Lab test results (continuous)
- Comorbidity indicators
- Geographic/socioeconomic factors
- Trial site (multi-site studies)

---

**Data Version:** v2.0-causal  
**Last Updated:** 2025-12-27  
**Status:** Production Quality
