"""
api/main.py
===========
PRODUCTION FASTAPI APPLICATION FOR DROPOUT PREDICTION

Serves LightGBM model with:
- Input validation
- Feature engineering
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

from prediction_logger import init_logger, get_logger
import config  # Import configuration

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Clinical Trial Dropout Prediction API",
    description="Predict patient dropout risk in clinical trials using LightGBM",
    version="1.0.0"
)

# Global model and preprocessor (loaded on startup)
model = None
preprocessor = None


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """
    API request schema for dropout prediction.
    
    All fields required for feature engineering.
    """
    patient_id: str = Field(..., description="Unique patient identifier", min_length=1)
    age: int = Field(
        ..., 
        ge=config.VALIDATION_RANGES["age"]["min"],
        le=config.VALIDATION_RANGES["age"]["max"],
        description=f"Patient age ({config.VALIDATION_RANGES['age']['min']}-{config.VALIDATION_RANGES['age']['max']})"
    )
    gender: Literal["Male", "Female", "Non-binary"] = Field(..., description="Patient gender")
    treatment_group: Literal["Active", "Control", "Placebo"] = Field(..., description="Treatment assignment")
    trial_phase: Literal["Phase I", "Phase II", "Phase III"] = Field(..., description="Current trial phase")
    days_in_trial: int = Field(..., gt=0, description="Days since enrollment")
    visits_completed: int = Field(..., ge=0, description="Total completed visits")
    last_visit_day: int = Field(..., ge=0, description="Day of last visit (0 if no visits)")
    adverse_events: int = Field(..., ge=0, description="Total adverse events reported")
    
    @validator('last_visit_day')
    def validate_last_visit(cls, v, values):
        """Ensure last_visit_day <= days_in_trial"""
        if 'days_in_trial' in values and v > values['days_in_trial']:
            raise ValueError(f"last_visit_day ({v}) cannot exceed days_in_trial ({values['days_in_trial']})")
        return v
    
    @validator('visits_completed')
    def validate_visits(cls, v, values):
        """Warn if no visits but patient still enrolled"""
        if v == 0:
            logger.warning(f"Patient has 0 visits completed")
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
    API response schema (USER-FACING ONLY).
    
    No internal metadata exposed.
    """
    patient_id: str
    dropout_prediction: int = Field(..., description="Binary prediction: 0 (retain) or 1 (dropout)")
    risk_level: str = Field(..., description="Risk stratification: Low/Moderate/High/Critical")
    recommended_action: str = Field(..., description="Suggested intervention action")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P-1234",
                "dropout_prediction": 1,
                "risk_level": "High",
                "recommended_action": "weekly_monitoring"
            }
        }


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform raw input into model features.
    
    Args:
        input_data: Raw patient data
        
    Returns:
        DataFrame with 9 engineered features
    """
    # Extract base features
    age = input_data['age']
    days_in_trial = input_data['days_in_trial']
    visits_completed = input_data['visits_completed']
    last_visit_day = input_data['last_visit_day']
    adverse_events = input_data['adverse_events']
    treatment_group = input_data['treatment_group']
    trial_phase = input_data['trial_phase']
    
    # Rate features (using config constants)
    visit_rate = visits_completed / (days_in_trial / config.EXPECTED_VISIT_INTERVAL_DAYS + config.EPSILON)
    adverse_event_rate = adverse_events / (days_in_trial + config.EPSILON)
    time_since_last_visit = days_in_trial - last_visit_day
    
    # Interaction features
    burden = adverse_event_rate * (1 - visit_rate)
    age_adverse_risk = (age / config.AGE_MAX) * adverse_event_rate
    
    # Domain features (from config)
    trial_phase_risk = config.TRIAL_PHASE_RISK_MAP[trial_phase]
    treatment_risk = config.TREATMENT_RISK_MAP[treatment_group]
    
    # Create DataFrame with expected feature order
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


def get_risk_level(probability: float) -> str:
    """
    Map probability to risk stratification level.
    
    Args:
        probability: Dropout probability (0-1)
        
    Returns:
        Risk level string
    """
    if probability >= config.RISK_THRESHOLDS["critical"]:
        return "Critical"
    elif probability >= config.RISK_THRESHOLDS["high"]:
        return "High"
    elif probability >= config.RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    else:
        return "Low"


def get_recommended_action(probability: float) -> str:
    """
    Map probability to recommended intervention.
    
    Args:
        probability: Dropout probability (0-1)
        
    Returns:
        Action recommendation string
    """
    if probability >= config.RISK_THRESHOLDS["critical"]:
        return "immediate_intervention"
    elif probability >= config.RISK_THRESHOLDS["high"]:
        return "weekly_monitoring"
    elif probability >= config.RISK_THRESHOLDS["moderate"]:
        return "biweekly_check"
    else:
        return "standard_protocol"


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model and initialize logging on startup"""
    global model, preprocessor
    
    logger.info("🚀 Starting API server...")
    logger.info(f"Configuration: {config.get_config_summary()}")
    
    # Load preprocessor
    if not config.PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {config.PREPROCESSOR_PATH}")
    
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    logger.info(f"✅ Loaded preprocessor: {config.PREPROCESSOR_PATH}")
    
    # Load model from MLflow or joblib
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        model_uri = f"models:/{config.MLFLOW_MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"✅ Loaded model from MLflow: {model_uri}")
    except Exception as e:
        logger.warning(f"Could not load from MLflow: {e}")
        # Fallback: load from file if available
        model_path = config.MODELS_DIR / f"lightgbm_dropout_{config.MODEL_VERSION}.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"✅ Loaded model from file: {model_path}")
        else:
            raise RuntimeError("Model not found in MLflow or local storage")
    
    # Initialize prediction logger (using config)
    init_logger(
        model_version=config.MODEL_VERSION,
        model_stage=config.MODEL_STAGE,
        decision_threshold=config.DECISION_THRESHOLD,
        log_file=str(config.PREDICTION_LOG_PATH)
    )
    logger.info("✅ Prediction logger initialized")
    logger.info(f"📊 Model: {config.MODEL_VERSION} | Stage: {config.MODEL_STAGE} | Threshold: {config.DECISION_THRESHOLD}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Clinical Trial Dropout Prediction API",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict dropout risk for a patient.
    
    Returns clean user-facing prediction (no internal metadata).
    Server-side logging captures full traceability.
    """
    start_time = time.time()
    
    try:
        # 1. Feature engineering
        features = engineer_features(request.dict())
        logger.debug(f"Engineered features for {request.patient_id}: {features.shape}")
        
        # 2. Preprocessing (StandardScaler)
        features_scaled = preprocessor.transform(features)
        
        # 3. Prediction
        probability = model.predict_proba(features_scaled)[0, 1]
        prediction = int(probability >= config.DECISION_THRESHOLD)
        
        # 4. Risk stratification
        risk_level = get_risk_level(probability)
        recommended_action = get_recommended_action(probability)
        
        # 5. SERVER-SIDE LOGGING (invisible to user)
        pred_logger = get_logger()
        pred_logger.log_prediction(
            patient_id=request.patient_id,
            input_data=request.dict(),
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            latency_ms=(time.time() - start_time) * 1000,
            metadata={
                "treatment_group": request.treatment_group,
                "trial_phase": request.trial_phase
            }
        )
        
        # 6. USER-FACING RESPONSE (clean, no metadata)
        return PredictionResponse(
            patient_id=request.patient_id,
            dropout_prediction=prediction,
            risk_level=risk_level,
            recommended_action=recommended_action
        )
        
    except Exception as e:
        # Log error
        pred_logger = get_logger()
        pred_logger.log_error(
            patient_id=request.patient_id,
            error_type=type(e).__name__,
            error_message=str(e),
            input_data=request.dict()
        )
        
        logger.error(f"Prediction error for {request.patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    Get session statistics (admin endpoint).
    
    Returns server-side metadata for monitoring.
    """
    try:
        pred_logger = get_logger()
        stats = pred_logger.get_session_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.LOG_LEVEL.lower()
    )
