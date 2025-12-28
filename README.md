# Clinical Trial Dropout Prediction System
**Production Branch**

This repository contains the production-ready MLOps pipeline for predicting patient dropout in clinical trials. It exposes a refined XGBoost model via a FastAPI interface with financial risk stratification.

## 🚀 Features (v1.1)
- **Production Model**: XGBoost (Fixed & Validated)
- **Risk Stratification**: 3-Tier Logic (Low, Medium, Critical)
- **Financial Logic**: Intervention cost estimation ($0.50 - $500.00)
- **Interfaces**: REST API (FastAPI) + Swagger UI

## 📂 Project Structure
```bash
├── api/                 # Production API (FastAPI)
├── src/                 # Core ML Logic (Training, Preprocessing)
├── models/              # Serialized Models (xgboost.pkl)
├── data/                # Data Assets (DVC tracked)
├── clean_and_train.py   # Full Pipeline Runner
└── Dockerfile           # Deployment Container
```

## 🛠️ Quick Start (Deployment)
1. **Run with Docker** (Recommended):
   ```bash
   docker-compose up --build
   ```
2. **Access API**: `http://localhost:8000/docs`

## 🧪 Experiments & Research
For experimental scripts, diagnosis logs, test JSONs, and detailed reports, switch to the **research** branch:
```bash
git checkout research
```

## ⚖️ Risk Logic
| Risk Level | Probability | Action | Cost |
| :--- | :--- | :--- | :--- |
| **Critical** | ≥ 80% | Retention Team | $500.00 |
| **Medium** | 40% - 79% | Nurse Consultation | $45.00 |
| **Low** | < 40% | SMS Alert | $0.50 |
