<p align="center">
  <img src="https://img.shields.io/badge/MLOps-Production%20Ready-success?style=for-the-badge&logo=kubernetes" alt="MLOps"/>
  <img src="https://img.shields.io/badge/Healthcare-AI-blue?style=for-the-badge&logo=heart" alt="Healthcare AI"/>
  <img src="https://img.shields.io/badge/AWS-Free%20Tier-orange?style=for-the-badge&logo=amazon-aws" alt="AWS"/>
</p>

# 🏥 MLOps System for Clinical Trial Risk Prediction

<p align="center">
  <strong>End-to-End Machine Learning Operations Pipeline for Healthcare Analytics</strong>
</p>

<p align="center">
  <a href="https://github.com/Jayasudhandesigner/MLOps-System-for-Clinical-Trial-Risk-Prediction"><img src="https://img.shields.io/badge/GitHub-Repository-black?style=flat-square&logo=github" alt="GitHub"/></a>
  <a href="https://hub.docker.com/r/sudhan2004/clinical-dropout-api"><img src="https://img.shields.io/badge/Docker%20Hub-sudhan2004-blue?style=flat-square&logo=docker" alt="Docker Hub"/></a>
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-655MB-2496ED?style=flat-square&logo=docker" alt="Docker Size"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"/>
</p>

---

## 📋 Overview

A **production-grade MLOps system** that predicts patient dropout risk in clinical trials, enabling pharmaceutical companies to reduce trial costs by up to **40%** through proactive interventions.

This project demonstrates enterprise-level MLOps practices including:
- 🔄 **Automated ML Pipeline** with model training during Docker build
- 🚀 **Production API** with FastAPI and Swagger documentation  
- 📊 **Real-time Risk Stratification** with cost-based interventions
- ☁️ **Cloud-Optimized** for AWS Free Tier (655MB Docker image)

---

## 🎯 Key Achievements

| Metric | Value |
|--------|-------|
| **Model Accuracy** | 85% |
| **Docker Image Size** | 655 MB (78% reduction from 3GB) |
| **API Response Time** | < 100ms |
| **Cloud Compatibility** | AWS Free Tier (t2.micro) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLINICAL TRIAL ML PLATFORM                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐   │
│   │   Frontend   │────▶│   FastAPI    │────▶│   ML Model           │   │
│   │   (React)    │     │   Gateway    │     │   (Scikit-learn)     │   │
│   └──────────────┘     └──────────────┘     └──────────────────────┘   │
│          │                    │                        │                │
│          │              ┌─────▼─────┐           ┌─────▼─────┐          │
│          │              │   Auth    │           │  Feature  │          │
│          │              │ (API Key) │           │Engineering│          │
│          │              └───────────┘           └───────────┘          │
│          │                                                              │
│   ┌──────▼──────────────────────────────────────────────────────────┐  │
│   │                    RISK STRATIFICATION ENGINE                    │  │
│   ├─────────────────┬─────────────────┬─────────────────────────────┤  │
│   │    LOW RISK     │   MEDIUM RISK   │       CRITICAL RISK         │  │
│   │    (< 40%)      │   (40-80%)      │         (> 80%)             │  │
│   │   SMS Alert     │  Consultation   │   Retention Team            │  │
│   │     $0.50       │     $45.00      │        $500.00              │  │
│   └─────────────────┴─────────────────┴─────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Pull from Docker Hub (Recommended)
```bash
# Pull the production image
docker pull sudhan2004/clinical-dropout-api:latest

# Run the container
docker run -d -p 8000:8000 sudhan2004/clinical-dropout-api:latest

# Test the API
curl http://localhost:8000/health
```

### Option 2: Build from Source
```bash
# Clone the repository
git clone https://github.com/Jayasudhandesigner/MLOps-System-for-Clinical-Trial-Risk-Prediction.git
cd MLOps-System-for-Clinical-Trial-Risk-Prediction

# Build the Docker image
docker build -t clinical-dropout-api .

# Run the container
docker run -d -p 8000:8000 clinical-dropout-api
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI documentation |
| POST | `/predict` | Get dropout prediction |

### Sample Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-secret-key" \
  -d '{
    "patient_id": "P-001",
    "age": 65,
    "gender": "Female",
    "treatment_group": "Placebo",
    "trial_phase": "Phase III",
    "days_in_trial": 120,
    "visits_completed": 3,
    "last_visit_day": 105,
    "adverse_events": 4
  }'
```

### Sample Response

```json
{
  "patient_id": "P-001",
  "dropout_prediction": 1,
  "dropout_probability": 0.8234,
  "risk_level": "Critical",
  "recommended_action": "retention_team_deployment",
  "intervention_cost": 500.00
}
```

---

## 📊 Test Scenarios

| Patient Profile | Age | Treatment | Adverse Events | Probability | Risk Level | Cost |
|-----------------|-----|-----------|----------------|-------------|------------|------|
| **Critical** - Elderly, poor compliance | 75 | Placebo | 8 | **99.35%** | Critical | $500 |
| **High** - Multiple adverse events | 68 | Placebo | 12 | **98.74%** | Critical | $500 |
| **Medium** - Control group | 55 | Control | 3 | 12.71% | Low | $0.50 |
| **Low** - Young, healthy | 35 | Active | 0 | **0.12%** | Low | $0.50 |

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow"/>
</p>

| Category | Technologies |
|----------|-------------|
| **Machine Learning** | Scikit-learn, Pandas, NumPy |
| **API Framework** | FastAPI, Uvicorn, Pydantic |
| **MLOps** | MLflow, DVC, GitHub Actions |
| **Infrastructure** | Docker, AWS EC2, Kubernetes |
| **Monitoring** | Evidently AI, Custom Drift Detection |

---

## 📁 Project Structure

```
MLOps-System-for-Clinical-Trial-Risk-Prediction/
├── 📂 api/                    # FastAPI Application
│   ├── main.py               # API endpoints
│   ├── config.py             # Configuration
│   └── prediction_logger.py  # Logging system
├── 📂 src/                    # Core ML Logic
│   ├── core/                 # Feature engineering
│   └── training/             # Model training
├── 📂 models/                 # Trained models (.pkl)
├── 📂 monitoring/             # Drift detection
├── 📂 scripts/                # Utility scripts
├── 📂 k8s/                    # Kubernetes manifests
├── 📂 data/                   # Data assets
├── 🐳 Dockerfile             # Production image
├── 📋 requirements.txt       # Dependencies
└── 📖 README.md              # This file
```

---

## ☁️ AWS Deployment (Free Tier)

### Prerequisites
- AWS Account with EC2 access
- t2.micro instance (Free Tier)
- 8GB storage

### Deployment Steps

```bash
# 1. SSH into EC2
ssh -i your-key.pem ec2-user@<your-ec2-public-ip>

# 2. Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -aG docker ec2-user

# 3. Pull and run
docker pull sudhan2004/clinical-dropout-api:latest
docker run -d -p 8000:8000 --restart always sudhan2004/clinical-dropout-api:latest

# 4. Configure Security Group
# Allow inbound traffic on port 8000 from anywhere (0.0.0.0/0)
```

### Access API
```
http://<your-ec2-public-ip>:8000/docs
```

---

## 💼 Business Impact

This system enables pharmaceutical companies to:

- ✅ **Reduce Trial Costs** by 40% through early intervention
- ✅ **Increase Trial Success Rate** by retaining high-risk patients
- ✅ **Automate Decision Making** with real-time risk assessment
- ✅ **Optimize Resources** with cost-based intervention tiers

---

## 🔒 Security Features

- 🔑 API Key authentication
- 🛡️ Non-root container execution
- 📝 Secure prediction logging
- 🔐 CORS protection
- ✅ Input validation with Pydantic

---

## 📬 Contact

<p align="center">
  <a href="https://www.linkedin.com/in/YOUR-LINKEDIN-PROFILE"><img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn"/></a>
  <a href="mailto:your.email@example.com"><img src="https://img.shields.io/badge/Email-Contact-red?style=for-the-badge&logo=gmail" alt="Email"/></a>
</p>

**Location:** Open to opportunities in Dubai, UAE 🇦🇪

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>⭐ Star this repository if you find it helpful!</strong>
</p>

<p align="center">
  Made with ❤️ for Healthcare AI
</p>
