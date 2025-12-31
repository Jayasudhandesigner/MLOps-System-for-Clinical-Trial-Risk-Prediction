"""
api/main.py
===========
PRODUCTION FASTAPI APPLICATION FOR DROPOUT PREDICTION

Serves XGBoost model (Fixed Version) with:
- Input validation
- Feature engineering (Direct features)
- Prediction logging (server-side traceability)
- Clean user-facing responses
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal, Dict, Any
import joblib
import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path

from api.prediction_logger import init_logger, get_logger
from api import config  # Import configuration

# Setup logging
logging.basicConfig(
    level=getattr(logging, "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Clinical Trial Dropout Prediction API",
    description="Predict patient dropout risk in clinical trials using XGBoost",
    version="1.1.0"
)

# Global model
model = None

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """API request schema for dropout prediction."""
    patient_id: str = Field(..., description="Unique patient identifier", min_length=1)
    age: int = Field(..., ge=18, le=100, description="Patient age")
    gender: Literal["Male", "Female", "Non-binary"] = Field(..., description="Patient gender")
    treatment_group: Literal["Active", "Control", "Placebo"] = Field(..., description="Treatment assignment")
    trial_phase: Literal["Phase I", "Phase II", "Phase III"] = Field(..., description="Current trial phase")
    days_in_trial: int = Field(..., gt=0, description="Days since enrollment")
    visits_completed: int = Field(..., ge=0, description="Total completed visits")
    last_visit_day: int = Field(..., ge=0, description="Day of last visit (0 if no visits)")
    adverse_events: int = Field(..., ge=0, description="Total adverse events reported")
    
    @validator('last_visit_day')
    def validate_last_visit(cls, v, values):
        if 'days_in_trial' in values and v > values['days_in_trial']:
            raise ValueError(f"last_visit_day ({v}) cannot exceed days_in_trial ({values['days_in_trial']})")
        return v
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    """API response schema (USER-FACING ONLY)."""
    patient_id: str
    dropout_prediction: int = Field(..., description="Binary prediction: 0 (retain) or 1 (dropout)")
    dropout_probability: float = Field(..., description="Probability score (0-1)")
    risk_level: str = Field(..., description="Risk stratification: Low/Medium/Critical")
    recommended_action: str = Field(..., description="Suggested intervention")
    intervention_cost: float = Field(..., description="Estimated cost of intervention ($)")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P-1234",
                "dropout_prediction": 1,
                "dropout_probability": 0.85,
                "risk_level": "Critical",
                "recommended_action": "retention_team_deployment",
                "intervention_cost": 500.00
            }
        }


# ============================================================================
# FEATURE ENGINEERING (MATCHING FIXED MODEL)
# ============================================================================

def engineer_features(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform raw input into model features.
    MUST MATCH: src/core/features.py + src/core/preprocess.py (v3_causal)
    
    Features (in order):
    - age, days_in_trial (numeric base)
    - visit_rate, adverse_event_rate, time_since_last_visit (rates)
    - burden, age_adverse_risk (interactions)
    - trial_phase_risk, treatment_risk (domain)
    """
    # Extract base features
    age = input_data['age']
    days_in_trial = input_data['days_in_trial']
    visits_completed = input_data['visits_completed']
    last_visit_day = input_data['last_visit_day']
    adverse_events = input_data['adverse_events']
    treatment_group = input_data['treatment_group']
    trial_phase = input_data['trial_phase']
    
    # === RATE FEATURES (from src/core/features.py) ===
    # Visit rate: compliance measure
    expected_visits = days_in_trial / 30 + 1
    visit_rate = visits_completed / expected_visits
    
    # Adverse event rate: normalized by time
    adverse_event_rate = adverse_events / (days_in_trial + 1)
    
    # Time since last visit: engagement gap
    time_since_last_visit = days_in_trial - last_visit_day
    
    # === INTERACTION FEATURES ===
    # Burden: compound pressure from adverse events + poor compliance
    burden = adverse_event_rate * (1 - visit_rate)
    
    # Age-adverse risk: older patients tolerate adverse events less
    age_adverse_risk = (age / 85) * adverse_event_rate
    
    # === DOMAIN FEATURES ===
    # Trial phase risk (ordinal)
    trial_phase_risk_map = {'Phase I': 0.2, 'Phase II': 0.5, 'Phase III': 0.8}
    trial_phase_risk = trial_phase_risk_map.get(trial_phase, 0.5)
    
    # Treatment group risk (ordinal)
    treatment_risk_map = {'Active': 0.1, 'Control': 0.3, 'Placebo': 0.4}
    treatment_risk = treatment_risk_map.get(treatment_group, 0.3)
    
    # Create DataFrame with EXACT feature order from preprocess.py
    # Order: NUMERIC_FEATURES + RATE_FEATURES + INTERACTION_FEATURES + DOMAIN_FEATURES
    features = pd.DataFrame([{
        'age': age,
        'days_in_trial': days_in_trial,
        'visit_rate': visit_rate,
        'adverse_event_rate': adverse_event_rate,
        'time_since_last_visit': time_since_last_visit,
        'burden': burden,
        'age_adverse_risk': age_adverse_risk,
        'trial_phase_risk': trial_phase_risk,
        'treatment_risk': treatment_risk
    }])
    
    return features


def get_risk_assessment(probability: float) -> Dict[str, Any]:
    """
    Determine risk level, action, and cost based on probability.
    
    Simplified 3-Tier Logic:
    - 0.80+ : Critical   ($500) - Retention Team
    - 0.40+ : Medium     ($45)  - Nurse/Doctor Consultation
    - <0.40 : Low        ($0.50) - Automated SMS/App Alert
    """
    if probability >= 0.80:
        return {
            "level": "Critical",
            "action": "retention_team_deployment",
            "cost": 500.00
        }
    elif probability >= 0.40:
        return {
            "level": "Medium",
            "action": "nurse_doctor_consultation",
            "cost": 45.00
        }
    else:
        return {
            "level": "Low",
            "action": "automated_sms_alert",
            "cost": 0.50
        }


# ============================================================================
# API ENDPOINTS
# ============================================================================

# Environment-based model path
import os
MODEL_PATH = os.getenv("MODEL_PATH", "models/production_model.pkl")

@app.on_event("startup")
async def startup_event():
    """Load model and initialize logging on startup (resilient)"""
    global model
    
    logger.info("🚀 Starting API server...")
    
    # Try environment variable path first, then fallbacks
    model_paths = [
        Path(MODEL_PATH),                          # From environment
        Path("models/production_model.pkl"),       # Production (Logistic)
        Path("models/logistic_fixed.pkl"),         # Logistic fallback
        Path("models/xgboost_fixed.pkl"),          # XGBoost fallback
        Path("models/lightgbm_fixed.pkl"),         # LightGBM fallback
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                logger.info(f"✅ Loaded model from: {model_path}")
                model_loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_path}: {e}")
    
    if not model_loaded:
        logger.error(f"⚠️ No model found. API started without model.")
        logger.error(f"   Tried: {[str(p) for p in model_paths]}")
        logger.error(f"   /predict will return 503 until model is available")
        # DON'T raise - let API start for health checks
    
    # Initialize prediction logger
    try:
        init_logger(
            model_version="v_fixed",
            model_stage="production",
            decision_threshold=0.5,
            log_file="logs/predictions.log"
        )
        logger.info("✅ Prediction logger initialized")
    except Exception as e:
        logger.warning(f"Prediction logger failed: {e}")
    
    logger.info("✅ API startup complete")


@app.get("/")
async def root():
    return {
        "message": "Clinical Trial Dropout Prediction API",
        "version": "2.0.0",
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check - always returns 200 (independent of model)"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    # Check if model is available
    if model is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Model not available. Service is starting up or model failed to load."
        )
    
    try:
        # 1. Feature engineering
        features = engineer_features(request.dict())
        
        # 2. Prediction
        probability = model.predict_proba(features)[0, 1]
        
        # Get threshold from config (default 0.5)
        # This allows K8s ConfigMap to control sensitivity
        threshold = float(os.getenv("DECISION_THRESHOLD", 0.5))
        prediction = int(probability >= threshold)
        
        # 3. Post-processing (New Risk Assessment with Cost)
        assessment = get_risk_assessment(probability)
        risk_level = assessment["level"]
        action = assessment["action"]
        cost = assessment["cost"]
        
        # 4. Log for internal tracing
        logger.info(f"Prediction: {prediction} (Prob: {probability:.3f}) - {risk_level} (${cost})")
        
        # 5. Log for monitoring/drift detection (Day 15)
        try:
            from monitoring.prediction_monitor import log_prediction
            log_prediction(
                input_data=request.dict(),
                output_data={
                    "dropout_prediction": prediction,
                    "dropout_probability": round(probability, 4),
                    "risk_level": risk_level,
                    "recommended_action": action,
                    "intervention_cost": cost
                }
            )
        except Exception as log_error:
            logger.warning(f"Monitoring log failed: {log_error}")
        
        return {
            "patient_id": request.patient_id,
            "dropout_prediction": prediction,
            "dropout_probability": round(probability, 4),
            "risk_level": risk_level,
            "recommended_action": action,
            "intervention_cost": cost
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

