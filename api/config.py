"""
api/config.py
=============
CONFIGURATION FILE - NO HARDCODING

All configurable parameters for the API.
"""

import os
from pathlib import Path
from typing import Dict

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_VERSION = os.getenv("MODEL_VERSION", "v3_causal")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.20"))

# ============================================================================
# FILE PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

PREPROCESSOR_PATH = DATA_DIR / "processed" / f"preprocessor_dropout_{MODEL_VERSION}.pkl"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
PREDICTION_LOG_PATH = LOGS_DIR / "predictions.jsonl"

# ============================================================================
# MLFLOW CONFIGURATION
# ============================================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
MLFLOW_MODEL_NAME = f"ClinicalTrialDropout_dropout_{MODEL_VERSION}"

# ============================================================================
# RISK STRATIFICATION THRESHOLDS
# ============================================================================

RISK_THRESHOLDS = {
    "critical": float(os.getenv("RISK_THRESHOLD_CRITICAL", "0.40")),
    "high": float(os.getenv("RISK_THRESHOLD_HIGH", "0.20")),
    "moderate": float(os.getenv("RISK_THRESHOLD_MODERATE", "0.10"))
}

# ============================================================================
# DOMAIN FEATURE MAPPINGS
# ============================================================================

TRIAL_PHASE_RISK_MAP: Dict[str, float] = {
    "Phase I": 0.2,
    "Phase II": 0.5,
    "Phase III": 0.8
}

TREATMENT_RISK_MAP: Dict[str, float] = {
    "Active": 0.1,
    "Control": 0.3,
    "Placebo": 0.4
}

# ============================================================================
# API SERVER CONFIGURATION
# ============================================================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================================
# VALIDATION RANGES
# ============================================================================

VALIDATION_RANGES = {
    "age": {"min": 18, "max": 85},
    "days_in_trial": {"min": 1, "max": 1000},
    "visits_completed": {"min": 0, "max": 100},
    "adverse_events": {"min": 0, "max": 50}
}

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Avoid division by zero
EPSILON = 1e-6

# Normalization constants
AGE_MAX = 85.0
EXPECTED_VISIT_INTERVAL_DAYS = 30

# ============================================================================
# ENVIRONMENT INFO
# ============================================================================

def get_config_summary() -> Dict:
    """Get configuration summary for logging/debugging"""
    return {
        "model_version": MODEL_VERSION,
        "model_stage": MODEL_STAGE,
        "decision_threshold": DECISION_THRESHOLD,
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "prediction_log": str(PREDICTION_LOG_PATH),
        "api_host": API_HOST,
        "api_port": API_PORT,
        "log_level": LOG_LEVEL
    }
