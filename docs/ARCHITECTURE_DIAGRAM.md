# System Architecture Diagram

This document illustrates the separation of concerns between the Consumer (React App), the Gateway (FastAPI), and the ML Core (MLflow/Retraining).

```text
┌───────────────────┐                  ┌───────────────────────────────────────────┐
│  React Frontend   │                  │             AI Backend Platform           │
│  (Vercel/Netlify) │                  │              (AWS EC2 / K8s)              │
│                   │                  │                                           │
│   [ Patient Form ]│                  │  ┌──────────────┐       ┌──────────────┐  │
│          │        │                  │  │              │       │              │  │
│          ▼        │  HTTPS + API Key │  │ FastAPI      │◄─────►│ MLflow       │  │
│   [ API Client ] ────────────────────┼─►│ Gateway      │       │ Registry     │  │
│                   │                  │  │ - Auth       │       │ - Versions   │  │
└───────────────────┘                  │  │ - Risk Logic │       │ - Artifacts  │  │
                                       │  └──────┬───────┘       └──────▲───────┘  │
                                       │         │                      │          │
                                       │         ▼                      │          │
                                       │  ┌──────────────┐       ┌──────┴───────┐  │
                                       │  │              │       │              │  │
                                       │  │ Prediction   │──────►│ CI/CD        │  │
                                       │  │ Logger       │ Drift │ Pipeline     │  │
                                       │  │ (Anonymized) │ Alert │(Auto-Retrain)│  │
                                       │  └──────────────┘       └──────────────┘  │
                                       └───────────────────────────────────────────┘
```

## Data Flow
1. **Frontend**: Collects patient data, validates inputs, sends secure JSON payload.
2. **FastAPI**:
   - Authenticates request (API Key).
   - Loads production model from Registry.
   - Engineers features (e.g., interaction terms).
   - Generates Probability + Risk Bucket (Low/Med/Critical).
   - Logs anonymous metadata for monitoring.
3. **Monitoring Loop**:
   - Evidently AI analyzes logs for **Drift**.
   - If Drift > 30% OR Recall < 55%: Triggers Retraining.
4. **CI/CD Pipeline**:
   - Retrains model on new data.
   - Compares Candidate vs Production.
   - If better → Auto-Promote to Registry.
