# Technical Documentation Index

**MLOps System for Clinical Trial Dropout Prediction**  
**Version:** 2.0-causal  
**Status:** Production Ready

---

## Overview

This repository contains a production-grade MLOps system for predicting patient dropout in clinical trials. The system implements causal data generation, advanced feature engineering, and experiment tracking to achieve 64% ROC-AUC with 72% recall.

---

## Documentation Structure

### Quick Start
- `README.md` - Project overview, installation, and basic usage
- `docs/QUICK_REFERENCE.md` - Command reference and metrics summary

### Technical Documentation
- `docs/architecture.md` - System architecture and design decisions
- `docs/data_contract.md` - Data schema and validation specifications
- `docs/feature_spec.md` - Feature engineering methodology
- `docs/OPTIMIZATION_GUIDE.md` - Implementation details of 6 core optimizations

### Development Guides
- `docs/COMPLETE_TUTORIAL.md` - End-to-end pipeline walkthrough
- `PROJECT_PROGRESS.md` - Development timeline and milestones
- `GIT_PUSH_COMPLETE.md` - Release notes for v2.0-causal

---

## For Developers

### Getting Started
```bash
# Clone and setup
git clone <repository-url>
cd MLOps-System-for-Clinical-Trial-Risk-Prediction
pip install -r requirements.txt

# Generate synthetic data
python data/synthetic_data_causal.py

# Run pipeline
python pipelines/local_pipeline.py

# View experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Core Components
- `src/core/` - Production pipeline (ingest → features → preprocess → train)
- `src/experiments/` - Research and development code
- `pipelines/` - End-to-end workflow orchestration
- `data/` - Data generation scripts

### Key Implementation Details
1. **Causal Data Generation** - Risk-based probability modeling
2. **Feature Engineering** - Rate normalization and interaction terms
3. **Class Balancing** - Triple-layer approach (stratification, weights, SMOTE)
4. **Experiment Tracking** - MLflow with feature versioning
5. **Model Training** - LogisticRegression, XGBoost, LightGBM

---

## Performance Metrics

| Model | CV ROC-AUC | Test ROC-AUC | Recall |
|-------|------------|--------------|--------|
| Logistic Regression | 0.698 | 0.643 | 0.720 |
| LightGBM | 0.643 | 0.618 | 0.700 |
| XGBoost | 0.648 | 0.604 | 0.680 |

**Baseline (random data):** 0.47 ROC-AUC  
**Current (causal data):** 0.64 ROC-AUC  
**Improvement:** +36%

---

## Repository Structure

```
MLOps/
├── src/core/           # Production code
├── src/experiments/    # Research code
├── pipelines/          # Workflow orchestration
├── data/               # Data generation
├── docs/               # Technical documentation
├── .gitignore          # Artifact exclusions
├── requirements.txt    # Dependencies
└── README.md           # Project overview
```

---

## Technical Stack

- **ML:** scikit-learn, XGBoost, LightGBM
- **Data:** pandas, numpy
- **Tracking:** MLflow
- **Environment:** Python 3.8+

---

## Version History

- **v2.0-causal** (2025-12-27) - Causal signal implementation, +36% performance
- **v0.1-baseline** - Initial implementation

---

**For detailed technical specifications, see individual documentation files.**
