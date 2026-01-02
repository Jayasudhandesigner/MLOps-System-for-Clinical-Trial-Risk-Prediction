"""
api/main.py
===========
PRODUCTION FASTAPI APPLICATION FOR DROPOUT PREDICTION

Version: 2.0.0 - Risk Band Consistency Update

Serves model with:
- Deterministic risk banding (not raw probabilities)
- Input validation
- Feature engineering (transform-only, no fitting)
- Prediction logging (server-side traceability)
- Clean user-facing responses

GUARANTEE: Same input → Same risk_band (within tolerance)
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
import time
import logging
import os
from pathlib import Path

from api.prediction_logger import init_logger, get_logger
from api.risk_bands import (
    get_risk_band, 
    get_risk_assessment, 
    compute_input_hash,
    validate_score_range,
    THRESHOLD_VERSION,
    RiskBand
)
from api import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# VERSION & MODEL METADATA
# ============================================================================

API_VERSION = "2.0.0"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v_fixed")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")

# FastAPI app
app = FastAPI(
    title="Clinical Trial Dropout Prediction API",
    description="Predict patient dropout risk with deterministic risk banding",
    version=API_VERSION
)

# ============================================================================
# SECURITY LAYER
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In real PROD, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_NAME = "x-api-key"
API_KEY_VALUE = os.getenv("API_KEY", "dev-secret-key")

if API_KEY_VALUE == "dev-secret-key":
    logger.warning("⚠️  Using INSECURE default API key. Set API_KEY env var in production!")

async def verify_api_key(x_api_key: str = Header(..., alias=API_KEY_NAME)):
    """Enforce API Key authentication on protected endpoints."""
    if x_api_key != API_KEY_VALUE:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key"
        )
    return x_api_key

# ============================================================================
# GLOBAL STATE
# ============================================================================

model = None
model_source = "none"  # Track where model was loaded from

# ============================================================================
# REQUEST/RESPONSE MODELS (UPDATED API CONTRACT)
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
    """
    API response schema - RISK BAND FOCUSED
    
    UI should rely on risk_band, NOT dropout_score for decision making.
    """
    patient_id: str
    dropout_score: float = Field(..., description="Raw model score (0-1)")
    risk_band: Literal["Low", "Medium", "High", "Critical"] = Field(..., description="Deterministic risk tier")
    recommended_action: str = Field(..., description="Suggested intervention")
    intervention_cost: float = Field(..., description="Estimated cost of intervention ($)")
    model_version: str = Field(..., description="Model version identifier")
    model_stage: str = Field(..., description="Model stage (production/staging)")
    model_source: str = Field(..., description="Model source (production/fallback)")
    threshold_version: str = Field(..., description="Risk threshold version")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P-1234",
                "dropout_score": 0.78,
                "risk_band": "Critical",
                "recommended_action": "retention_team_deployment",
                "intervention_cost": 500.00,
                "model_version": "v_fixed",
                "model_stage": "production",
                "model_source": "production",
                "threshold_version": "1.0.0"
            }
        }


# ============================================================================
# FEATURE ENGINEERING (TRANSFORM ONLY - NO FIT)
# ============================================================================

def engineer_features(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform raw input into model features.
    
    CRITICAL: This uses .transform() logic only - no fitting.
    Feature order and calculations MUST match training pipeline.
    
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
    expected_visits = days_in_trial / 30 + 1
    visit_rate = visits_completed / expected_visits
    adverse_event_rate = adverse_events / (days_in_trial + 1)
    time_since_last_visit = days_in_trial - last_visit_day
    
    # === INTERACTION FEATURES ===
    burden = adverse_event_rate * (1 - visit_rate)
    age_adverse_risk = (age / 85) * adverse_event_rate
    
    # === DOMAIN FEATURES (FIXED MAPPINGS - FROM config.py) ===
    trial_phase_risk_map = {'Phase I': 0.2, 'Phase II': 0.5, 'Phase III': 0.8}
    trial_phase_risk = trial_phase_risk_map.get(trial_phase, 0.5)
    
    treatment_risk_map = {'Active': 0.1, 'Control': 0.3, 'Placebo': 0.4}
    treatment_risk = treatment_risk_map.get(treatment_group, 0.3)
    
    # Create DataFrame with EXACT feature order
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


# ============================================================================
# API ENDPOINTS
# ============================================================================

MODEL_PATH = os.getenv("MODEL_PATH", "models/production_model.pkl")

@app.on_event("startup")
async def startup_event():
    """Load model and initialize logging on startup."""
    global model, model_source
    
    logger.info(f"🚀 Starting API server v{API_VERSION}...")
    logger.info(f"📊 Threshold version: {THRESHOLD_VERSION}")
    
    # Model loading priority (production first, then fallbacks)
    model_paths = [
        (Path(MODEL_PATH), "production"),
        (Path("models/production_model.pkl"), "production"),
        (Path("models/logistic_fixed.pkl"), "fallback"),
        (Path("models/xgboost_fixed.pkl"), "fallback"),
        (Path("models/lightgbm_fixed.pkl"), "fallback"),
    ]
    
    model_loaded = False
    for model_path, source in model_paths:
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                model_source = source
                logger.info(f"✅ Loaded model from: {model_path} (source: {source})")
                
                # Warn if using fallback in production
                if source == "fallback":
                    logger.warning(f"⚠️  Using FALLBACK model. Production model not found.")
                
                model_loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_path}: {e}")
    
    if not model_loaded:
        model_source = "none"
        logger.error(f"⚠️ No model found. API started without model.")
        logger.error(f"   /predict will return 503 until model is available")
    
    # Initialize prediction logger
    try:
        init_logger(
            model_version=MODEL_VERSION,
            model_stage=MODEL_STAGE,
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
        "version": API_VERSION,
        "status": "healthy",
        "model_loaded": model is not None,
        "model_source": model_source,
        "threshold_version": THRESHOLD_VERSION
    }


@app.get("/health")
async def health_check():
    """Health check - always returns 200 (independent of model)."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_source": model_source,
        "model_path": MODEL_PATH,
        "threshold_version": THRESHOLD_VERSION
    }


@app.get("/config")
async def get_config():
    """Get current configuration (for debugging)."""
    from api.risk_bands import get_threshold_info
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "model_stage": MODEL_STAGE,
        "model_source": model_source,
        "thresholds": get_threshold_info()
    }


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(request: PredictionRequest):
    """
    Make dropout prediction with deterministic risk banding.
    
    GUARANTEE: Same input → Same risk_band
    UI should rely on risk_band, NOT dropout_score.
    """
    start_time = time.time()
    
    # Check if model is available
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Service is starting up or model failed to load."
        )
    
    try:
        # 1. Compute input hash for logging/debugging
        input_dict = request.dict()
        input_hash = compute_input_hash(input_dict)
        
        # 2. Feature engineering (transform only)
        features = engineer_features(input_dict)
        
        # 3. Model prediction
        raw_score = model.predict_proba(features)[0, 1]
        raw_score = validate_score_range(raw_score)
        
        # 4. RISK BANDING (core consistency guarantee)
        assessment = get_risk_assessment(raw_score)
        risk_band = assessment["risk_band"]
        action = assessment["recommended_action"]
        cost = assessment["intervention_cost"]
        
        # 5. Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # 6. Log for internal tracing
        logger.info(
            f"Prediction: {risk_band} | "
            f"Score: {raw_score:.4f} | "
            f"Hash: {input_hash} | "
            f"Source: {model_source} | "
            f"Latency: {latency_ms:.1f}ms"
        )
        
        # 7. Log for monitoring/drift detection
        try:
            from monitoring.prediction_monitor import log_prediction
            log_prediction(
                input_data=input_dict,
                output_data={
                    "dropout_score": round(raw_score, 4),
                    "risk_band": risk_band,
                    "recommended_action": action,
                    "intervention_cost": cost,
                    "input_hash": input_hash,
                    "model_version": MODEL_VERSION,
                    "model_source": model_source,
                    "threshold_version": THRESHOLD_VERSION,
                    "latency_ms": round(latency_ms, 2)
                }
            )
        except Exception as log_error:
            logger.warning(f"Monitoring log failed: {log_error}")
        
        # 8. Return response (UI relies on risk_band, not dropout_score)
        return {
            "patient_id": request.patient_id,
            "dropout_score": round(raw_score, 4),
            "risk_band": risk_band,
            "recommended_action": action,
            "intervention_cost": cost,
            "model_version": MODEL_VERSION,
            "model_stage": MODEL_STAGE,
            "model_source": model_source,
            "threshold_version": THRESHOLD_VERSION
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BACKWARD COMPATIBILITY ENDPOINT (DEPRECATED)
# ============================================================================

@app.post("/predict/legacy", dependencies=[Depends(verify_api_key)])
async def predict_legacy(request: PredictionRequest):
    """
    Legacy endpoint for backward compatibility.
    
    DEPRECATED: Use /predict instead.
    Returns old response format with dropout_prediction and dropout_probability.
    """
    logger.warning("⚠️ Legacy endpoint called - consider migrating to /predict")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        features = engineer_features(request.dict())
        raw_score = model.predict_proba(features)[0, 1]
        raw_score = validate_score_range(raw_score)
        
        threshold = float(os.getenv("DECISION_THRESHOLD", 0.5))
        prediction = int(raw_score >= threshold)
        
        assessment = get_risk_assessment(raw_score)
        
        return {
            "patient_id": request.patient_id,
            "dropout_prediction": prediction,
            "dropout_probability": round(raw_score, 4),
            "risk_level": assessment["risk_band"],
            "recommended_action": assessment["recommended_action"],
            "intervention_cost": assessment["intervention_cost"]
        }
        
    except Exception as e:
        logger.error(f"Legacy prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
