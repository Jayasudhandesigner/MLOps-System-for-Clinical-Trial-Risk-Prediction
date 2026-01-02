"""
api/risk_bands.py
==================
PRODUCTION RISK BANDING MODULE

Core principle: Deterministic risk stratification decoupled from raw model scores.
This ensures consistent risk tiers across local and production environments.

THRESHOLD VERSION: 1.0.0
"""

from enum import Enum
from typing import Dict, Any
import hashlib
import json

# ============================================================================
# RISK BAND CONFIGURATION (Version-controlled)
# ============================================================================

THRESHOLD_VERSION = "1.0.0"

class RiskBand(str, Enum):
    """Risk band enumeration for clinical dropout prediction."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


# Deterministic thresholds - DO NOT CHANGE without version bump
RISK_THRESHOLDS = {
    RiskBand.CRITICAL: 0.75,  # >= 0.75 → Critical
    RiskBand.HIGH: 0.55,      # >= 0.55 → High
    RiskBand.MEDIUM: 0.35,    # >= 0.35 → Medium
    # < 0.35 → Low
}

# Recommended actions per risk band
RISK_ACTIONS = {
    RiskBand.CRITICAL: {
        "action": "retention_team_deployment",
        "description": "Immediate retention team intervention required",
        "cost": 500.00,
        "priority": 1
    },
    RiskBand.HIGH: {
        "action": "nurse_doctor_consultation",
        "description": "Schedule urgent consultation with care team",
        "cost": 150.00,
        "priority": 2
    },
    RiskBand.MEDIUM: {
        "action": "proactive_outreach",
        "description": "Proactive phone call or telehealth check-in",
        "cost": 45.00,
        "priority": 3
    },
    RiskBand.LOW: {
        "action": "automated_sms_alert",
        "description": "Standard monitoring with automated reminders",
        "cost": 0.50,
        "priority": 4
    }
}


# ============================================================================
# CORE RISK BANDING FUNCTION
# ============================================================================

def get_risk_band(raw_score: float) -> RiskBand:
    """
    Deterministic risk banding function.
    
    Maps raw model score to a clinical risk band.
    Same input → Same output (guaranteed).
    
    Args:
        raw_score: Model probability output (0.0 - 1.0)
        
    Returns:
        RiskBand enum value
        
    Thresholds:
        CRITICAL >= 0.75
        HIGH >= 0.55
        MEDIUM >= 0.35
        LOW < 0.35
    """
    # Clamp score to valid range
    score = max(0.0, min(1.0, float(raw_score)))
    
    if score >= RISK_THRESHOLDS[RiskBand.CRITICAL]:
        return RiskBand.CRITICAL
    elif score >= RISK_THRESHOLDS[RiskBand.HIGH]:
        return RiskBand.HIGH
    elif score >= RISK_THRESHOLDS[RiskBand.MEDIUM]:
        return RiskBand.MEDIUM
    else:
        return RiskBand.LOW


def get_risk_assessment(raw_score: float) -> Dict[str, Any]:
    """
    Complete risk assessment with band, action, and metadata.
    
    Args:
        raw_score: Model probability output (0.0 - 1.0)
        
    Returns:
        Dictionary with:
        - risk_band: Risk tier (Low/Medium/High/Critical)
        - recommended_action: Suggested intervention
        - intervention_cost: Estimated cost ($)
        - action_description: Human-readable description
        - priority: Action priority (1=highest)
        - threshold_version: Version of threshold logic
    """
    band = get_risk_band(raw_score)
    action_info = RISK_ACTIONS[band]
    
    return {
        "risk_band": band.value,
        "recommended_action": action_info["action"],
        "intervention_cost": action_info["cost"],
        "action_description": action_info["description"],
        "priority": action_info["priority"],
        "threshold_version": THRESHOLD_VERSION
    }


# ============================================================================
# CONSISTENCY UTILITIES
# ============================================================================

def compute_input_hash(input_data: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of input data for logging/debugging.
    
    Args:
        input_data: Dictionary of input features
        
    Returns:
        SHA256 hash (first 16 chars)
    """
    # Sort keys for deterministic serialization
    serialized = json.dumps(input_data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def validate_score_range(raw_score: float) -> float:
    """
    Validate and clamp raw score to valid probability range.
    
    Args:
        raw_score: Model output
        
    Returns:
        Clamped score in [0.0, 1.0]
    """
    if raw_score < 0.0 or raw_score > 1.0:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Raw score {raw_score} outside [0,1] range - clamping")
    
    return max(0.0, min(1.0, float(raw_score)))


# ============================================================================
# VERSION INFO
# ============================================================================

def get_threshold_info() -> Dict[str, Any]:
    """Get current threshold configuration for debugging/logging."""
    return {
        "version": THRESHOLD_VERSION,
        "thresholds": {
            "critical": RISK_THRESHOLDS[RiskBand.CRITICAL],
            "high": RISK_THRESHOLDS[RiskBand.HIGH],
            "medium": RISK_THRESHOLDS[RiskBand.MEDIUM],
            "low": 0.0
        },
        "bands": [band.value for band in RiskBand]
    }


if __name__ == "__main__":
    # Test risk banding
    test_scores = [0.10, 0.35, 0.55, 0.75, 0.90]
    print("Risk Banding Test:")
    print("-" * 50)
    for score in test_scores:
        band = get_risk_band(score)
        assessment = get_risk_assessment(score)
        print(f"Score: {score:.2f} → {band.value} (${assessment['intervention_cost']})")
    print("-" * 50)
    print(f"Threshold Version: {THRESHOLD_VERSION}")
