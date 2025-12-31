"""
monitoring/should_retrain.py
=============================
THE DECISION BRAIN - Determines if retraining is required

This script evaluates drift metrics against policy thresholds
and outputs a simple decision: RETRAIN_REQUIRED=true/false

Usage:
    python monitoring/should_retrain.py
    
Exit Codes:
    0 - Decision made successfully
    1 - Error (missing files, invalid data)
"""

import yaml
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

# Paths
POLICY_PATH = Path("config/retraining_policy.yaml")
DRIFT_PATH = Path("monitoring/drift_summary.json")
LAST_RETRAIN_PATH = Path("monitoring/last_retrain.json")
AUDIT_LOG_PATH = Path("logs/retraining_audit.jsonl")


def load_policy() -> Dict[str, Any]:
    """Load retraining policy configuration."""
    if not POLICY_PATH.exists():
        print(f"ERROR: Policy file not found: {POLICY_PATH}", file=sys.stderr)
        sys.exit(1)
    
    with open(POLICY_PATH) as f:
        return yaml.safe_load(f)


def load_drift_summary() -> Dict[str, Any]:
    """Load current drift metrics."""
    if not DRIFT_PATH.exists():
        # No drift data = no retraining needed (first run)
        return {
            "data_drift_pct": 0.0,
            "prediction_shift": 0.0,
            "recall": 1.0,
            "roc_auc": 1.0,
            "timestamp": datetime.now().isoformat()
        }
    
    with open(DRIFT_PATH) as f:
        return json.load(f)


def check_cooldown(policy: Dict[str, Any]) -> Tuple[bool, int]:
    """Check if we're still in cooldown period."""
    cooldown_days = policy.get("cooldown_days", 7)
    
    if not LAST_RETRAIN_PATH.exists():
        return False, 0  # No previous retrain, no cooldown
    
    with open(LAST_RETRAIN_PATH) as f:
        last_retrain = json.load(f)
    
    last_date = datetime.fromisoformat(last_retrain["timestamp"])
    days_since = (datetime.now() - last_date).days
    
    if days_since < cooldown_days:
        return True, cooldown_days - days_since
    
    return False, 0


def should_retrain(drift: Dict[str, Any], policy: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Evaluate if retraining is required based on drift and policy.
    
    Returns:
        Tuple[bool, str]: (should_retrain, reason)
    """
    triggers = policy.get("retraining_triggers", {})
    
    # Check data drift
    data_drift_threshold = triggers.get("data_drift", {}).get("max_drifted_features_pct", 0.30)
    data_drift_pct = drift.get("data_drift_pct", 0.0)
    if data_drift_pct > data_drift_threshold:
        return True, f"data_drift ({data_drift_pct:.1%} > {data_drift_threshold:.1%})"
    
    # Check prediction drift
    pred_drift_threshold = triggers.get("prediction_drift", {}).get("max_distribution_shift", 0.20)
    prediction_shift = drift.get("prediction_shift", 0.0)
    if prediction_shift > pred_drift_threshold:
        return True, f"prediction_drift ({prediction_shift:.1%} > {pred_drift_threshold:.1%})"
    
    # Check performance decay - recall
    min_recall = triggers.get("performance", {}).get("min_recall", 0.55)
    current_recall = drift.get("recall", 1.0)
    if current_recall < min_recall:
        return True, f"performance_decay (recall {current_recall:.3f} < {min_recall})"
    
    # Check performance decay - ROC-AUC
    min_roc_auc = triggers.get("performance", {}).get("min_roc_auc", 0.58)
    current_roc_auc = drift.get("roc_auc", 1.0)
    if current_roc_auc < min_roc_auc:
        return True, f"performance_decay (roc_auc {current_roc_auc:.3f} < {min_roc_auc})"
    
    return False, "no_trigger"


def log_decision(decision: bool, reason: str, drift: Dict[str, Any], cooldown: bool = False):
    """Log the retraining decision for audit purposes."""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": "retrain_decision",
        "decision": "retrain" if decision else "skip",
        "reason": reason,
        "cooldown_active": cooldown,
        "metrics": {
            "data_drift_pct": drift.get("data_drift_pct", 0.0),
            "prediction_shift": drift.get("prediction_shift", 0.0),
            "recall": drift.get("recall", None),
            "roc_auc": drift.get("roc_auc", None)
        }
    }
    
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    """Main decision logic."""
    print("=" * 60)
    print("🧠 RETRAINING DECISION ENGINE")
    print("=" * 60)
    
    # Load inputs
    policy = load_policy()
    drift = load_drift_summary()
    
    print(f"\n📊 Current Metrics:")
    print(f"   Data Drift:       {drift.get('data_drift_pct', 0):.1%}")
    print(f"   Prediction Shift: {drift.get('prediction_shift', 0):.1%}")
    print(f"   Recall:           {drift.get('recall', 'N/A')}")
    print(f"   ROC-AUC:          {drift.get('roc_auc', 'N/A')}")
    
    # Check cooldown
    in_cooldown, days_remaining = check_cooldown(policy)
    if in_cooldown:
        print(f"\n⏳ COOLDOWN ACTIVE: {days_remaining} days remaining")
        print("\nRETRAIN_REQUIRED=false")
        log_decision(False, "cooldown_active", drift, cooldown=True)
        return
    
    # Make decision
    retrain, reason = should_retrain(drift, policy)
    
    print(f"\n📋 Policy Thresholds:")
    triggers = policy.get("retraining_triggers", {})
    print(f"   Max Data Drift:   {triggers.get('data_drift', {}).get('max_drifted_features_pct', 0.30):.1%}")
    print(f"   Max Pred Shift:   {triggers.get('prediction_drift', {}).get('max_distribution_shift', 0.20):.1%}")
    print(f"   Min Recall:       {triggers.get('performance', {}).get('min_recall', 0.55)}")
    
    print(f"\n{'=' * 60}")
    if retrain:
        print(f"🔴 RETRAIN REQUIRED")
        print(f"   Reason: {reason}")
    else:
        print(f"🟢 NO RETRAINING NEEDED")
        print(f"   Status: All metrics within acceptable thresholds")
    print(f"{'=' * 60}")
    
    # Output for CI/CD
    print(f"\nRETRAIN_REQUIRED={'true' if retrain else 'false'}")
    
    # Log decision
    log_decision(retrain, reason, drift)


if __name__ == "__main__":
    main()
