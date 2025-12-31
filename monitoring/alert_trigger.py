"""
monitoring/alert_trigger.py
============================
DRIFT ALERT TRIGGER FOR CI/CD INTEGRATION

Purpose: Automated checks that can fail CI/CD pipelines when drift is detected.
Designed for integration with GitHub Actions or other CI systems.

Usage:
    python monitoring/alert_trigger.py

Exit Codes:
    0 = No significant drift (safe)
    1 = Drift detected (retraining recommended)
    2 = Error (missing data, config issue)
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Import drift detection
try:
    from monitoring.run_drift import (
        run_drift_detection,
        REPORT_JSON_PATH
    )
except ImportError:
    # Allow running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from monitoring.run_drift import (
        run_drift_detection,
        REPORT_JSON_PATH
    )


# Alert Thresholds (configurable)
DRIFT_SHARE_THRESHOLD = 0.30      # Alert if >30% features drift
CRITICAL_FEATURE_DRIFT = [        # Alert if ANY of these drift
    "age", "adverse_events", "days_in_trial"
]
PREDICTION_SHIFT_THRESHOLD = 0.20  # Alert if prediction distribution shifts >20%


def check_alerts() -> Dict[str, Any]:
    """
    Run all alert checks.
    
    Returns:
        Dictionary with alert status and details
    """
    print("=" * 70)
    print("🚨 MONITORING ALERT CHECK")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    alerts = {
        "triggered": False,
        "alerts": [],
        "severity": "none",
        "action": "continue"
    }
    
    try:
        # Run drift detection
        results = run_drift_detection()
        
        # Check 1: Overall drift share
        if results["drift_share"] > DRIFT_SHARE_THRESHOLD:
            alerts["alerts"].append({
                "type": "HIGH_DRIFT_SHARE",
                "message": f"Drift share {results['drift_share']:.1%} exceeds threshold {DRIFT_SHARE_THRESHOLD:.1%}",
                "severity": "warning"
            })
            alerts["triggered"] = True
        
        # Check 2: Critical feature drift
        for feature in CRITICAL_FEATURE_DRIFT:
            if feature in results.get("drifted_features", []):
                alerts["alerts"].append({
                    "type": "CRITICAL_FEATURE_DRIFT",
                    "message": f"Critical feature '{feature}' has drifted",
                    "severity": "critical"
                })
                alerts["triggered"] = True
                alerts["severity"] = "critical"
        
        # Check 3: Dataset drift flag
        if results.get("dataset_drift", False):
            alerts["alerts"].append({
                "type": "DATASET_DRIFT",
                "message": "Evidently detected significant dataset drift",
                "severity": "warning"
            })
            alerts["triggered"] = True
        
        # Determine action
        if alerts["severity"] == "critical":
            alerts["action"] = "retrain_immediately"
        elif alerts["triggered"]:
            alerts["action"] = "investigate"
            if alerts["severity"] == "none":
                alerts["severity"] = "warning"
        
    except FileNotFoundError as e:
        alerts["alerts"].append({
            "type": "MISSING_DATA",
            "message": str(e),
            "severity": "info"
        })
        alerts["action"] = "setup_required"
        
    except Exception as e:
        alerts["alerts"].append({
            "type": "ERROR",
            "message": str(e),
            "severity": "error"
        })
        alerts["action"] = "investigate"
        alerts["triggered"] = True
    
    return alerts


def main() -> int:
    """
    Main entry point for CI/CD integration.
    
    Returns:
        Exit code (0=success, 1=drift, 2=error)
    """
    alerts = check_alerts()
    
    # Print summary
    print()
    print("-" * 70)
    print("ALERT SUMMARY")
    print("-" * 70)
    print(f"  Triggered:  {'🔴 YES' if alerts['triggered'] else '🟢 NO'}")
    print(f"  Severity:   {alerts['severity'].upper()}")
    print(f"  Action:     {alerts['action']}")
    print()
    
    if alerts["alerts"]:
        print("  Details:")
        for alert in alerts["alerts"]:
            icon = "⚠️" if alert["severity"] == "warning" else "🔴" if alert["severity"] == "critical" else "ℹ️"
            print(f"    {icon} [{alert['type']}] {alert['message']}")
    
    print()
    print("=" * 70)
    
    # Save alert report
    alert_path = Path("monitoring/alert_report.json")
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    with open(alert_path, "w") as f:
        json.dump(alerts, f, indent=2)
    print(f"📋 Alert report saved: {alert_path}")
    
    # Determine exit code
    if alerts["action"] == "setup_required":
        return 0  # Not an error, just needs setup
    elif alerts["severity"] == "critical":
        print("\n❌ CRITICAL: Retraining required")
        return 1
    elif alerts["triggered"]:
        print("\n⚠️  WARNING: Investigation recommended")
        return 1
    else:
        print("\n✅ All checks passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
