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
    Matches logic in force_train_save.py
    """
    # Extract base features
    age = input_data['age']
    days_in_trial = input_data['days_in_trial']
    visits_completed = input_data['visits_completed']
    last_visit_day = input_data['last_visit_day']
    adverse_events = input_data['adverse_events']
    treatment_group = input_data['treatment_group']
    trial_phase = input_data['trial_phase']
    
    # 1. Visit Compliance
    expected_visits = days_in_trial / 30
    visit_compliance = visits_completed / (expected_visits + 0.1)
    
    # 2. Time since last visit
    time_since_last_visit = days_in_trial - last_visit_day
    
    # 3. Adverse Rate
    adverse_rate = adverse_events / (days_in_trial + 1)
    
    # 4. Encodings
    phase_map = {'Phase I': 1, 'Phase II': 2, 'Phase III': 3}
    treatment_map = {'Active': 1, 'Control': 2, 'Placebo': 3}
    
    # Create DataFrame with EXACT feature order from training
    features = pd.DataFrame([{
        'age': age,
        'days_in_trial': days_in_trial,
        'visits_completed': visits_completed,
        'visit_compliance': visit_compliance,
        'time_since_last_visit': time_since_last_visit,
        'adverse_events': adverse_events,
        'adverse_rate': adverse_rate,
        'trial_phase_encoded': phase_map.get(trial_phase, 0),
        'treatment_encoded': treatment_map.get(treatment_group, 0)
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

@app.on_event("startup")
async def startup_event():
    """Load model and initialize logging on startup"""
    global model
    
    logger.info("🚀 Starting API server...")
    
    # Load FIXED model
    model_path = Path("models/xgboost_fixed.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
        logger.info(f"✅ Loaded FIXED model from file: {model_path}")
    else:
        logger.error(f"❌ Model not found at {model_path}")
        raise RuntimeError("Model file missing")
    
    # Initialize logger
    init_logger(
        model_version="v_fixed",
        model_stage="production",
        decision_threshold=0.5,
        log_file="logs/predictions.log"
    )
    logger.info("✅ Prediction logger initialized")


@app.get("/")
async def root():
    return {
        "message": "Clinical Trial Dropout Prediction API (Fixed Version)",
        "version": "1.1.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    try:
        # 1. Feature engineering
        features = engineer_features(request.dict())
        
        # 2. Prediction (No scaler needed for this tree model)
        probability = model.predict_proba(features)[0, 1]
        prediction = int(probability >= 0.5)  # Threshold 0.5
        
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

