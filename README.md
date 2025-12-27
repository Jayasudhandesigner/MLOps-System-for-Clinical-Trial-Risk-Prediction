# Clinical Trial Dropout Prediction - Production MLOps System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-green.svg)](https://mlflow.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.64-success.svg)]()

Production-grade machine learning system for predicting patient dropout in clinical trials using causal inference and advanced feature engineering.

---

## Key Results

- **Performance:** ROC-AUC 0.643, Recall 0.720
- **Improvement:** +36% over baseline (0.47 → 0.64)
- **Clinical Impact:** Identifies 72% of at-risk patients for early intervention

---

## Quick Start

```bash
# Setup
git clone https://github.com/Jayasudhandesigner/MLOps-System-for-Clinical-Trial-Risk-Prediction.git
cd MLOps-System-for-Clinical-Trial-Risk-Prediction
pip install -r requirements.txt

# Generate causal data
python data/synthetic_data_causal.py

# Run production pipeline
python pipelines/local_pipeline.py

# View experiment results
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Runtime:** ~3 minutes | **Output:** MLflow UI at http://localhost:5000

---

## System Architecture

```
data/raw/clinical_trials.csv (causal generation)
            ↓
    [1] Data Ingestion → src/core/ingest.py
            ↓
    [2] Feature Engineering → src/core/features.py
        • Rate features (temporal normalization)
        • Interaction features (compound effects)
        • Domain encoding (risk mapping)
            ↓
    [3] Preprocessing → src/core/preprocess.py
        • StandardScaler (mean=0, std=1)
        • Feature versioning (v3_causal)
            ↓
    [4] Model Training → src/core/train.py
        • Logistic Regression, XGBoost, LightGBM
        • Class balancing (stratified, weighted)
        • 5-fold cross-validation
            ↓
    [5] Experiment Tracking → MLflow
        • Parameter & metric logging
        • Model registry & versioning
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [01_PROGRESS.md](docs/01_PROGRESS.md) | Development timeline (Days 1-8) |
| [02_ARCHITECTURE.md](docs/02_ARCHITECTURE.md) | System design & component specs |
| [03_DATA.md](docs/03_DATA.md) | Data schema & causal generation |
| [04_ML_MODEL.md](docs/04_ML_MODEL.md) | Model architecture & evaluation |
| [05_OPTIMIZATION_RESULTS.md](docs/05_OPTIMIZATION_RESULTS.md) | Performance & findings |

---

## Repository Structure

```
MLOps/
├── data/
│   └── synthetic_data_causal.py    # Causal data generation
│
├── src/core/                        # Production pipeline
│   ├── ingest.py                   # Data validation
│   ├── features.py                 # Feature engineering
│   ├── preprocess.py               # Scaling & versioning
│   └── train.py                    # Model training
│
├── pipelines/
│   └── local_pipeline.py           # End-to-end workflow
│
├── docs/                            # Technical documentation
│   ├── 01_PROGRESS.md
│   ├── 02_ARCHITECTURE.md
│   ├── 03_DATA.md
│   ├── 04_ML_MODEL.md
│   └── 05_OPTIMIZATION_RESULTS.md
│
├── .gitignore                       # Artifact exclusions
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## Technical Highlights

### Causal Data Generation
Dropout probability based on weighted risk score:
```python
risk = 0.35×adverse_rate + 0.30×(1-visit_rate) + 0.20×phase_risk
dropout = sigmoid(risk)
```

### Feature Engineering
- **Rate Features:** `visit_rate = visits / expected` (temporal normalization)
- **Interaction Features:** `burden = adverse_rate × (1 - visit_rate)` (compound effects)
- **Domain Encoding:** Trial phase & treatment risk mapping

### Class Imbalance Handling
- Stratified train/test splits (maintains 24% dropout distribution)
- Class weights (`class_weight='balanced'`, `scale_pos_weight`)
- Optional SMOTE oversampling

### Model Performance

| Model | CV ROC-AUC | Test ROC-AUC | Recall |
|-------|------------|--------------|--------|
| **Logistic Regression** | **0.698** | **0.643** | **0.720** |
| XGBoost | 0.648 | 0.604 | 0.680 |
| LightGBM | 0.643 | 0.618 | 0.700 |

**Winner:** Logistic Regression (linear separability from causal features)

---

## Requirements

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

## Reproducibility

All randomness controlled with `random_state=42`:
- Data generation: `np.random.seed(42)`
- Train/test split: `random_state=42`
- Model training: `random_state=42`
- Cross-validation: `random_state=42`

**Result:** Identical ROC-AUC 0.64 on every run

---

## Data Versioning

This project uses DVC to version raw and processed datasets.
Each model version can be traced back to an exact dataset snapshot.

```bash
# Pull versioned data
dvc pull

# Data is tracked but not stored in git
data/processed.dvc    # Metadata (in git)
data/processed/       # Actual files (in DVC cache)
```

---

## Key Findings

1. **Causal Data Quality Determines Success:** Random → 0.47, Causal → 0.64 (+36%)
2. **Feature Engineering > Model Complexity:** Good features + simple model wins
3. **Interaction Features Critical:** burden feature adds +7% ROC-AUC
4. **Class Balancing Essential:** Recall improves from 0.12 → 0.72 (6×)
5. **Linear Models Win on Causal Data:** Logistic Regression outperforms XGBoost

---

## Production Deployment

### Model Inference
```python
import joblib

# Load artifacts
model = joblib.load('models/logistic_regression_v2.pkl')
preprocessor = joblib.load('models/preprocessor_v3_causal.pkl')

# Predict on new patient
new_patient = pd.DataFrame([{...}])  # Patient features
X = preprocessor.transform(new_patient)
dropout_prob = model.predict_proba(X)[0, 1]

# High-risk threshold: 0.35 (vs default 0.5)
high_risk = dropout_prob > 0.35  # 85% recall, 55% precision
```

### Monitoring
- ROC-AUC should stay > 0.60
- Recall should stay > 0.70
- Feature drift monitoring (KL divergence)
- Retrain if performance degrades

---

## Future Enhancements

- [ ] Survival analysis (Cox Proportional Hazards)
- [ ] Time-to-event prediction
- [ ] Real-time API (FastAPI)
- [ ] Docker containerization
- [ ] Drift detection (Evidently AI)
- [ ] CI/CD pipeline (GitHub Actions)

---

## License

MIT License

---

## Contact

**Repository:** https://github.com/Jayasudhandesigner/MLOps-System-for-Clinical-Trial-Risk-Prediction  
**Version:** v2.0-causal  
**Status:** Production Ready
