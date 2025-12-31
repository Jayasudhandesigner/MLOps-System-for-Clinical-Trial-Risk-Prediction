"""
monitoring/run_drift.py
========================
DRIFT DETECTION REPORT GENERATOR

Purpose: Compare live predictions against reference data to detect:
- Data Drift (input feature changes)
- Prediction Drift (model output changes)

Uses Evidently AI for industry-standard drift detection.

Usage:
    python monitoring/run_drift.py

Output:
    monitoring/drift_report.html
    monitoring/drift_report.json (for CI/CD integration)
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import (
        DatasetDriftMetric,
        DataDriftTable,
        ColumnDriftMetric
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("⚠️  Evidently not installed. Run: pip install evidently")


# Paths
REFERENCE_PATH = "monitoring/reference.csv"
PREDICTIONS_PATH = "monitoring/live_predictions.jsonl"
REPORT_HTML_PATH = "monitoring/drift_report.html"
REPORT_JSON_PATH = "monitoring/drift_report.json"

# Feature columns to monitor
NUMERIC_FEATURES = [
    "age", "days_in_trial", "visits_completed", 
    "adverse_events", "last_visit_day"
]

CATEGORICAL_FEATURES = [
    "trial_phase", "treatment_group", "gender"
]

# Prediction columns to monitor
PREDICTION_COLUMNS = [
    "dropout_probability", "dropout_prediction", "risk_level"
]


def load_reference_data() -> pd.DataFrame:
    """Load reference dataset."""
    if not Path(REFERENCE_PATH).exists():
        raise FileNotFoundError(
            f"Reference data not found at {REFERENCE_PATH}. "
            "Run: python monitoring/build_reference.py"
        )
    return pd.read_csv(REFERENCE_PATH)


def load_current_data() -> pd.DataFrame:
    """Load current predictions from JSONL log."""
    if not Path(PREDICTIONS_PATH).exists():
        raise FileNotFoundError(
            f"No predictions found at {PREDICTIONS_PATH}. "
            "Make some predictions via the API first."
        )
    
    # Read JSONL
    records = []
    with open(PREDICTIONS_PATH, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        raise ValueError("Prediction log is empty")
    
    # Flatten input and output into single rows
    current_df = pd.json_normalize(records)
    
    # Rename columns to match reference format
    # input.age -> age, output.dropout_probability -> dropout_probability
    column_mapping = {}
    for col in current_df.columns:
        if col.startswith("input."):
            column_mapping[col] = col.replace("input.", "")
        elif col.startswith("output."):
            column_mapping[col] = col.replace("output.", "")
    
    current_df = current_df.rename(columns=column_mapping)
    
    return current_df


def prepare_for_comparison(
    reference: pd.DataFrame, 
    current: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare datasets for drift comparison.
    Align columns between reference and current data.
    """
    # Find common numeric columns
    common_numeric = [c for c in NUMERIC_FEATURES if c in reference.columns and c in current.columns]
    common_categorical = [c for c in CATEGORICAL_FEATURES if c in reference.columns and c in current.columns]
    
    # Add prediction columns if available
    pred_cols = [c for c in PREDICTION_COLUMNS if c in current.columns]
    
    # For reference, we need to simulate prediction columns
    if "dropout_probability" not in reference.columns:
        # Create synthetic prediction columns based on dropout
        reference["dropout_probability"] = reference["dropout"].apply(
            lambda x: 0.7 if x == 1 else 0.3
        )
        reference["dropout_prediction"] = reference["dropout"]
        reference["risk_level"] = reference["dropout_probability"].apply(
            lambda p: "Critical" if p >= 0.8 else ("Medium" if p >= 0.4 else "Low")
        )
    
    all_cols = common_numeric + common_categorical + pred_cols
    
    ref_aligned = reference[all_cols].copy() if all(c in reference.columns for c in all_cols) else reference[common_numeric + common_categorical].copy()
    cur_aligned = current[all_cols].copy() if all(c in current.columns for c in all_cols) else current[common_numeric + common_categorical].copy()
    
    return ref_aligned, cur_aligned


def run_drift_detection() -> Dict[str, Any]:
    """
    Run drift detection comparing reference to current data.
    
    Returns:
        Dictionary with drift detection results
    """
    print("=" * 70)
    print("🔍 DRIFT DETECTION ANALYSIS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    if not EVIDENTLY_AVAILABLE:
        raise ImportError("Evidently not available. Install with: pip install evidently")
    
    # Load data
    print("📁 Loading data...")
    reference = load_reference_data()
    current = load_current_data()
    
    print(f"   Reference: {len(reference)} records")
    print(f"   Current:   {len(current)} records")
    print()
    
    # Prepare for comparison
    ref_aligned, cur_aligned = prepare_for_comparison(reference, current)
    
    print(f"   Aligned columns: {list(ref_aligned.columns)}")
    print()
    
    # Build Evidently report
    print("🔬 Running drift analysis...")
    
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable()
    ])
    
    report.run(
        reference_data=ref_aligned,
        current_data=cur_aligned
    )
    
    # Save HTML report
    Path(REPORT_HTML_PATH).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(REPORT_HTML_PATH)
    print(f"📊 Saved HTML report: {REPORT_HTML_PATH}")
    
    # Extract results
    report_dict = report.as_dict()
    
    # Save JSON for CI/CD
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    print(f"📋 Saved JSON report: {REPORT_JSON_PATH}")
    
    # Parse results
    results = parse_drift_results(report_dict)
    
    # Print summary
    print()
    print("-" * 70)
    print("DRIFT DETECTION RESULTS")
    print("-" * 70)
    print(f"  Dataset Drift Detected: {'⚠️  YES' if results['dataset_drift'] else '✅ NO'}")
    print(f"  Drifted Features:       {results['drifted_count']}/{results['total_features']}")
    print(f"  Drift Share:            {results['drift_share']:.1%}")
    print()
    
    if results['drifted_features']:
        print("  Drifted Features:")
        for feat in results['drifted_features']:
            print(f"    • {feat}")
    
    print()
    print("=" * 70)
    
    return results


def parse_drift_results(report_dict: Dict) -> Dict[str, Any]:
    """Parse Evidently report dictionary into actionable results."""
    results = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "drifted_count": 0,
        "total_features": 0,
        "drifted_features": []
    }
    
    try:
        for metric in report_dict.get("metrics", []):
            metric_id = metric.get("metric", "")
            result = metric.get("result", {})
            
            if "DatasetDriftMetric" in metric_id:
                results["dataset_drift"] = result.get("dataset_drift", False)
                results["drift_share"] = result.get("drift_share", 0.0)
                results["drifted_count"] = result.get("number_of_drifted_columns", 0)
                results["total_features"] = result.get("number_of_columns", 0)
            
            if "DataDriftTable" in metric_id:
                drift_by_columns = result.get("drift_by_columns", {})
                for col, col_result in drift_by_columns.items():
                    if col_result.get("drift_detected", False):
                        results["drifted_features"].append(col)
    
    except Exception as e:
        print(f"⚠️  Error parsing results: {e}")
    
    return results


def check_drift_threshold(
    drift_share_threshold: float = 0.30,
    raise_on_drift: bool = False
) -> bool:
    """
    Check if drift exceeds threshold.
    
    Args:
        drift_share_threshold: Maximum acceptable drift share (0.30 = 30%)
        raise_on_drift: If True, raise RuntimeError on drift
    
    Returns:
        True if drift detected above threshold
    """
    results = run_drift_detection()
    
    drift_detected = results["drift_share"] > drift_share_threshold
    
    print()
    if drift_detected:
        msg = f"⚠️  DRIFT ALERT: {results['drift_share']:.1%} features drifted (threshold: {drift_share_threshold:.1%})"
        print(msg)
        if raise_on_drift:
            raise RuntimeError(msg + " — retraining recommended")
    else:
        print(f"✅ Drift within acceptable bounds ({results['drift_share']:.1%} < {drift_share_threshold:.1%})")
    
    return drift_detected


if __name__ == "__main__":
    try:
        check_drift_threshold(drift_share_threshold=0.30, raise_on_drift=False)
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print()
        print("To run drift detection:")
        print("  1. Build reference: python monitoring/build_reference.py")
        print("  2. Make predictions: POST to /predict endpoint")
        print("  3. Re-run this script")
    except Exception as e:
        print(f"❌ Error: {e}")
