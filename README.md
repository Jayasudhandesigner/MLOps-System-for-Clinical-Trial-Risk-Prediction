# Clinical Trial Dropout Prediction - MLOps System

Production-grade machine learning system for predicting patient dropout in clinical trials using causal feature engineering and advanced modeling techniques.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-green.svg)](https://mlflow.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.64-success.svg)]()

---

## Overview

Clinical trials experience significant patient dropout rates (15-30%), resulting in substantial financial losses ($2.6B annually) and trial delays (12-18 months). This system implements machine learning models to predict dropout risk, enabling early intervention.

### Key Features

- **Causal Data Generation:** Risk-based probability modeling for learnable signal creation
- **Advanced Feature Engineering:** Rate normalization, interaction terms, and domain knowledge encoding
- **Class Imbalance Handling:** Multi-layer approach including SMOTE, class weighting, and stratified sampling
- **Experiment Tracking:** MLflow integration with feature versioning and model registry
- **Production Architecture:** Clean separation of core pipeline and experimental code

### Performance

| Model | CV ROC-AUC | Test ROC-AUC | Recall |
|-------|------------|--------------|--------|
| Logistic Regression | 0.698 | 0.643 | 0.720 |
| LightGBM | 0.643 | 0.618 | 0.700 |
| XGBoost | 0.648 | 0.604 | 0.680 |

**Baseline (random features):** 0.47 ROC-AUC  
**Current (causal features):** 0.64 ROC-AUC  
**Improvement:** +36%

---

## System Architecture

```
Data Flow Pipeline:
──────────────────

data/raw/clinical_trials.csv (causal generation)
            ↓
    [1. Data Ingestion & Validation]
    src/core/ingest.py
            ↓
    [2. Feature Engineering]
    src/core/features.py
    • Rate features: visit_rate, adverse_event_rate
    • Interaction features: burden = adverse_rate × (1 - visit_rate)
    • Domain encoding: trial_phase_risk, treatment_risk
            ↓
    [3. Preprocessing]
    src/core/preprocess.py
    • StandardScaler normalization
    • Feature versioning (v3_causal)
            ↓
    [4. Model Training]
    src/core/train.py
    • Class balancing: stratified splits + class_weight
    • Cross-validation: 5-fold StratifiedKFold
    • Models: LogisticRegression, XGBoost, LightGBM
            ↓
    [5. Experiment Tracking]
    MLflow Registry
    • Version control
    • Metric logging
    • Model artifacts
```

---

## Installation

### Requirements

- Python 3.8+
- pip

### Setup

```bash
git clone https://github.com/Jayasudhandesigner/MLOps-System-for-Clinical-Trial-Risk-Prediction.git
cd MLOps-System-for-Clinical-Trial-Risk-Prediction
pip install -r requirements.txt
```

### Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
imbalanced-learn>=0.11.0
mlflow>=2.8.0
joblib>=1.3.0
```

---

## Usage

### 1. Generate Synthetic Data

```bash
python data/synthetic_data_causal.py
```

Generates 1000 synthetic patient records with causal dropout patterns based on risk scoring:

```python
risk_score = (
    0.35 * adverse_event_rate +
    0.30 * (1 - visit_rate) +
    0.20 * phase_risk +
    0.10 * treatment_risk
)
```

### 2. Run Pipeline

```bash
python pipelines/local_pipeline.py
```

Executes end-to-end workflow:
- Data preprocessing with feature engineering
- Model training for 3 algorithms
- Experiment logging to MLflow

Expected runtime: ~3 minutes

### 3. View Results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Access MLflow UI at `http://localhost:5000`

---

## Repository Structure

```
MLOps/
│
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── synthetic_data_causal.py      # Data generation
│
├── src/
│   ├── core/                         # Production pipeline
│   │   ├── ingest.py                 # Data validation
│   │   ├── features.py               # Feature engineering
│   │   ├── preprocess.py             # Transformation pipeline
│   │   └── train.py                  # Model training
│   │
│   └── experiments/                  # Research code
│       ├── train_optimized.py
│       ├── train_all_targets.py
│       └── compare_models.py
│
├── pipelines/
│   └── local_pipeline.py             # Workflow orchestration
│
├── docs/
│   ├── START_HERE.md                 # Documentation index
│   ├── architecture.md               # System design
│   ├── data_contract.md              # Data specifications
│   ├── feature_spec.md               # Feature details
│   └── OPTIMIZATION_GUIDE.md         # Technical deep-dive
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technical Implementation

### Causal Data Generation

Dropout probability is determined by a risk score combining multiple factors:

- **Adverse Event Rate:** `adverse_events / days_in_trial`
- **Visit Compliance:** `visits_completed / expected_visits`
- **Trial Phase Risk:** Ordinal encoding (Phase I: 0.2, Phase II: 0.5, Phase III: 0.8)
- **Treatment Risk:** Placebo/Control groups have higher dropout probability

### Feature Engineering

**Rate-Based Features (Temporal Normalization):**
```python
visit_rate = visits_completed / (days_in_trial / 30 + 1)
adverse_event_rate = adverse_events / (days_in_trial + 1)
time_since_last_visit = days_in_trial - last_visit_day
```

**Interaction Features (Compound Effects):**
```python
burden = adverse_event_rate × (1 - visit_rate)
age_adverse_risk = (age / 85) × adverse_event_rate
```

**Domain Knowledge Encoding:**
```python
trial_phase_risk = {'Phase I': 0.2, 'Phase II': 0.5, 'Phase III': 0.8}
treatment_risk = {'Active': 0.1, 'Control': 0.3, 'Placebo': 0.4}
```

### Class Imbalance Handling

**Triple-Layer Approach:**

1. **Stratified Splitting:** Maintains class distribution in train/test sets
2. **Class Weights:** Models weight minority class errors higher
   - Logistic Regression: `class_weight='balanced'`
   - XGBoost: `scale_pos_weight` = majority_count / minority_count
3. **SMOTE (Optional):** Synthetic minority oversampling

### Model Training

**Cross-Validation:**
- 5-fold StratifiedKFold
- Metrics: ROC-AUC, Precision, Recall, F1-Score

**Models:**
- **Logistic Regression:** Baseline linear model with StandardScaler pipeline
- **XGBoost:** Gradient boosting with scale_pos_weight
- **LightGBM:** Fast gradient boosting with class_weight

### Experiment Tracking

MLflow logs all runs with:
- **Parameters:** feature_version, model_type, target, hyperparameters
- **Metrics:** cv_roc_auc, test_roc_auc, recall, precision, f1_score
- **Artifacts:** Trained model, preprocessor pipeline

Feature versioning enables comparison:
- v1_counts: Raw count features (ROC-AUC 0.52)
- v2_rates: Rate-based features (ROC-AUC 0.63)
- v3_causal: Rates + interactions + domain (ROC-AUC 0.65)

---

## Configuration

### Multiple Dropout Targets

System supports 4 dropout prediction targets:

| Target | Definition | Use Case |
|--------|-----------|----------|
| `dropout` | Binary dropout indicator | General risk assessment |
| `early_dropout` | Dropout < 90 days | Onboarding intervention |
| `late_dropout` | Dropout ≥ 90 days | Long-term retention |
| `dropout_30_days` | Dropout ≤ 30 days | Critical early warning |

Configure in `pipelines/local_pipeline.py`:

```python
run_pipeline(
    target="dropout",
    feature_version="v3_causal",
    model_type="xgboost"
)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/START_HERE.md` | Documentation index and quick reference |
| `docs/architecture.md` | System architecture and design decisions |
| `docs/data_contract.md` | Data schema and validation specifications |
| `docs/feature_spec.md` | Feature engineering methodology |
| `docs/OPTIMIZATION_GUIDE.md` | Implementation details and optimizations |

---

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Features

1. Implement in `src/core/features.py`
2. Update `feature_version` in `src/core/preprocess.py`
3. Train models with new version
4. Compare in MLflow UI

### Contributing

See development workflow in `docs/architecture.md`

---

## Performance Metrics

### ROC-AUC Progression

| Stage | ROC-AUC | Change |
|-------|---------|--------|
| Random features | 0.47 | Baseline |
| Count features | 0.52 | +11% |
| Rate features | 0.63 | +21% |
| Causal features | 0.65 | +38% |

### Model Comparison

Best model: **Logistic Regression**
- Simpler models perform better on linearly separable causal data
- Rate-based features create linear decision boundaries
- Complex models (XGBoost, LightGBM) show slight overfitting

---

## Technical Stack

- **ML Frameworks:** scikit-learn, XGBoost, LightGBM
- **Data Processing:** pandas, numpy
- **Imbalance Handling:** imbalanced-learn (SMOTE)
- **Experiment Tracking:** MLflow
- **Serialization:** joblib

---

## Version History

### v2.0-causal (2025-12-27)
- Causal data generation with risk-based probability
- Rate-based feature engineering
- Interaction features (burden, age_adverse_risk)
- Domain knowledge encoding
- Triple-layer class balancing
- MLflow feature versioning
- Performance: ROC-AUC 0.64 (+36% vs baseline)

### v0.1-baseline
- Initial implementation
- Basic synthetic data generation
- Count-based features
- Single model training

---

## License

MIT License

---

## Contact

**Repository:** https://github.com/Jayasudhandesigner/MLOps-System-for-Clinical-Trial-Risk-Prediction

**Version:** v2.0-causal  
**Status:** Production Ready
