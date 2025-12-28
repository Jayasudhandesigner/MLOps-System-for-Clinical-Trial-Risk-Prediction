"""
api/prediction_logger.py
========================
SERVER-SIDE PREDICTION LOGGING FOR TRACEABILITY

Logs every prediction with model metadata for:
- Audit trail
- Model governance
- Debugging
- A/B testing
- Performance monitoring

User-facing responses DO NOT include this metadata.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import uuid

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Logs prediction metadata for traceability without exposing to users.
    
    Log format (JSON lines):
    {
        "session_id": "uuid",
        "timestamp": "ISO8601",
        "model_version": "v3_causal",
        "model_stage": "production",
        "decision_threshold": 0.20,
        "input_hash": "sha256",
        "prediction": 1,
        "probability": 0.78,
        "risk_level": "High",
        "patient_id": "P-1234",
        "latency_ms": 45
    }
    """
    
    def __init__(
        self,
        log_file: str = "logs/predictions.jsonl",
        model_version: str = "v3_causal",
        model_stage: str = "production",
        decision_threshold: float = 0.20
    ):
        """
        Initialize prediction logger.
        
        Args:
            log_file: Path to JSON lines log file
            model_version: Feature version identifier
            model_stage: Model stage (staging/production)
            decision_threshold: Threshold used for binary prediction
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Model metadata (SERVER-SIDE ONLY)
        self.model_version = model_version
        self.model_stage = model_stage
        self.decision_threshold = decision_threshold
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        
        logger.info(f"PredictionLogger initialized:")
        logger.info(f"  Session ID: {self.session_id}")
        logger.info(f"  Model: {model_version} ({model_stage})")
        logger.info(f"  Threshold: {decision_threshold}")
        logger.info(f"  Log file: {self.log_file}")
    
    def _hash_input(self, input_data: Dict[str, Any]) -> str:
        """
        Create hash of input data for deduplication/debugging.
        
        Args:
            input_data: Raw input dictionary
            
        Returns:
            SHA256 hash (first 16 chars)
        """
        # Sort keys for consistent hashing
        input_str = json.dumps(input_data, sort_keys=True)
        hash_obj = hashlib.sha256(input_str.encode())
        return hash_obj.hexdigest()[:16]
    
    def log_prediction(
        self,
        patient_id: str,
        input_data: Dict[str, Any],
        prediction: int,
        probability: float,
        risk_level: str,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log prediction with full metadata (SERVER-SIDE ONLY).
        
        Args:
            patient_id: Patient identifier
            input_data: Raw input features
            prediction: Binary prediction (0 or 1)
            probability: Dropout probability
            risk_level: Risk stratification level
            latency_ms: Prediction latency in milliseconds
            metadata: Additional metadata to log
        """
        log_entry = {
            # Session tracking
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            
            # Model metadata (INTERNAL ONLY)
            "model_version": self.model_version,
            "model_stage": self.model_stage,
            "decision_threshold": self.decision_threshold,
            
            # Input
            "patient_id": patient_id,
            "input_hash": self._hash_input(input_data),
            
            # Output
            "prediction": prediction,
            "probability": round(probability, 6),
            "risk_level": risk_level,
            
            # Performance
            "latency_ms": round(latency_ms, 2) if latency_ms else None,
            
            # Additional metadata
            **(metadata or {})
        }
        
        # Write as JSON line (append mode)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.debug(f"Logged prediction: {patient_id} → {prediction} (P={probability:.4f})")
    
    def log_error(
        self,
        patient_id: str,
        error_type: str,
        error_message: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log prediction errors for debugging.
        
        Args:
            patient_id: Patient identifier
            error_type: Error classification
            error_message: Error description
            input_data: Input that caused error (if available)
        """
        log_entry = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_version": self.model_version,
            "patient_id": patient_id,
            "error_type": error_type,
            "error_message": error_message,
            "input_hash": self._hash_input(input_data) if input_data else None
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.error(f"Logged error: {patient_id} → {error_type}: {error_message}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics for current session.
        
        Returns:
            Dictionary with session statistics
        """
        if not self.log_file.exists():
            return {
                "session_id": self.session_id,
                "total_predictions": 0
            }
        
        # Count predictions in this session
        predictions = []
        with open(self.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("session_id") == self.session_id and "prediction" in entry:
                    predictions.append(entry)
        
        if not predictions:
            return {
                "session_id": self.session_id,
                "total_predictions": 0
            }
        
        # Calculate stats
        total = len(predictions)
        positive_preds = sum(1 for p in predictions if p["prediction"] == 1)
        avg_prob = sum(p["probability"] for p in predictions) / total
        avg_latency = sum(p.get("latency_ms", 0) for p in predictions if p.get("latency_ms")) / total
        
        return {
            "session_id": self.session_id,
            "model_version": self.model_version,
            "total_predictions": total,
            "positive_predictions": positive_preds,
            "positive_rate": round(positive_preds / total, 4),
            "avg_probability": round(avg_prob, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "threshold": self.decision_threshold
        }


# Global logger instance (initialized by API on startup)
_prediction_logger: Optional[PredictionLogger] = None


def init_logger(
    model_version: str = "v3_causal",
    model_stage: str = "production",
    decision_threshold: float = 0.20,
    log_file: str = "logs/predictions.jsonl"
) -> PredictionLogger:
    """
    Initialize global prediction logger.
    
    Call this once on API startup.
    
    Args:
        model_version: Feature version
        model_stage: Model stage
        decision_threshold: Binary decision threshold
        log_file: Path to log file
        
    Returns:
        Initialized PredictionLogger
    """
    global _prediction_logger
    _prediction_logger = PredictionLogger(
        log_file=log_file,
        model_version=model_version,
        model_stage=model_stage,
        decision_threshold=decision_threshold
    )
    return _prediction_logger


def get_logger() -> PredictionLogger:
    """
    Get global prediction logger.
    
    Returns:
        Global PredictionLogger instance
        
    Raises:
        RuntimeError: If logger not initialized
    """
    if _prediction_logger is None:
        raise RuntimeError("PredictionLogger not initialized. Call init_logger() first.")
    return _prediction_logger


# Example usage in API endpoint
"""
from api.prediction_logger import get_logger

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    # Make prediction
    prediction, probability = model.predict(request.dict())
    
    # Log prediction (SERVER-SIDE ONLY)
    logger = get_logger()
    logger.log_prediction(
        patient_id=request.patient_id,
        input_data=request.dict(),
        prediction=prediction,
        probability=probability,
        risk_level=get_risk_level(probability),
        latency_ms=(time.time() - start_time) * 1000
    )
    
    # User-facing response (NO METADATA)
    return {
        "patient_id": request.patient_id,
        "dropout_prediction": prediction,
        "risk_level": get_risk_level(probability)
    }
"""
