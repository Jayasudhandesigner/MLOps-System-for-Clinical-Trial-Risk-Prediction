"""
monitoring/prediction_monitor.py
=================================
LIVE PREDICTION LOGGING FOR DRIFT DETECTION

Purpose: Log all predictions to enable monitoring.
Creates a JSONL stream of predictions for analysis.

Usage:
    from monitoring.prediction_monitor import log_prediction
    log_prediction(input_data, output_data)

Output:
    monitoring/live_predictions.jsonl
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import threading

# Thread-safe lock for file writing
_write_lock = threading.Lock()

# Default log path
DEFAULT_LOG_PATH = "monitoring/live_predictions.jsonl"


def log_prediction(
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    log_path: str = DEFAULT_LOG_PATH
) -> None:
    """
    Log a prediction to JSONL file for monitoring.
    
    Args:
        input_data: Raw input features (from API request)
        output_data: Prediction output (probability, risk level, etc.)
        log_path: Path to JSONL log file
    
    Format:
        {"timestamp": "...", "input": {...}, "output": {...}}
    """
    # Build record
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "output": output_data
    }
    
    # Ensure directory exists
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Thread-safe write
    with _write_lock:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def get_recent_predictions(
    log_path: str = DEFAULT_LOG_PATH,
    n: int = 100
) -> list:
    """
    Get the most recent N predictions.
    
    Args:
        log_path: Path to JSONL log file
        n: Number of recent predictions to return
    
    Returns:
        List of prediction records
    """
    log_file = Path(log_path)
    if not log_file.exists():
        return []
    
    predictions = []
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    return predictions[-n:]


def get_prediction_count(log_path: str = DEFAULT_LOG_PATH) -> int:
    """Get total number of logged predictions."""
    log_file = Path(log_path)
    if not log_file.exists():
        return 0
    
    with open(log_path, "r") as f:
        return sum(1 for line in f if line.strip())


def clear_predictions(log_path: str = DEFAULT_LOG_PATH) -> bool:
    """Clear prediction log (use with caution)."""
    log_file = Path(log_path)
    if log_file.exists():
        log_file.unlink()
        return True
    return False
