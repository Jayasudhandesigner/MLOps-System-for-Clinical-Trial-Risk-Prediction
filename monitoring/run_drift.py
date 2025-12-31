"""
monitoring/run_drift.py
========================
DRIFT DETECTION REPORT GENERATOR

Purpose: Compare live predictions against reference data to detect:
- Data Drift (input feature changes)
- Prediction Drift (model output changes)

Uses Evidently AI for industry-standard drift detection.
Includes fallback statistical methods if Evidently is unavailable.

Usage:
    python monitoring/run_drift.py

Output:
    monitoring/drift_report.html
    monitoring/drift_report.json (for CI/CD integration)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import warnings

# Suppress warnings during import
warnings.filterwarnings('ignore')

# Try importing Evidently with graceful fallback
EVIDENTLY_AVAILABLE = False
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import DatasetDriftMetric, DataDriftTable
    EVIDENTLY_AVAILABLE = True
except (ImportError, TypeError, KeyError) as e:
    print(f"⚠️  Evidently import issue: {type(e).__name__}")
    print("   Using fallback statistical drift detection")


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
    
    records = []
    with open(PREDICTIONS_PATH, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        raise ValueError("Prediction log is empty")
    
    current_df = pd.json_normalize(records)
    
    # Rename columns: input.age -> age, output.dropout_probability -> dropout_probability
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
    """Prepare datasets for drift comparison."""
    common_numeric = [c for c in NUMERIC_FEATURES if c in reference.columns and c in current.columns]
    common_categorical = [c for c in CATEGORICAL_FEATURES if c in reference.columns and c in current.columns]
    pred_cols = [c for c in PREDICTION_COLUMNS if c in current.columns]
    
    # Add synthetic prediction columns to reference if needed
    if "dropout_probability" not in reference.columns and "dropout" in reference.columns:
        reference = reference.copy()
        reference["dropout_probability"] = reference["dropout"].apply(
            lambda x: 0.7 if x == 1 else 0.3
        )
        reference["dropout_prediction"] = reference["dropout"]
        reference["risk_level"] = reference["dropout_probability"].apply(
            lambda p: "Critical" if p >= 0.8 else ("Medium" if p >= 0.4 else "Low")
        )
    
    all_cols = common_numeric + common_categorical + pred_cols
    available_ref_cols = [c for c in all_cols if c in reference.columns]
    available_cur_cols = [c for c in all_cols if c in current.columns]
    common_cols = list(set(available_ref_cols) & set(available_cur_cols))
    
    return reference[common_cols].copy(), current[common_cols].copy()


# =============================================================================
# FALLBACK: Statistical Drift Detection (when Evidently not available)
# =============================================================================

def ks_test_drift(ref_col: pd.Series, cur_col: pd.Series, threshold: float = 0.05) -> Tuple[bool, float]:
    """Kolmogorov-Smirnov test for numerical drift."""
    from scipy import stats
    
    ref_clean = ref_col.dropna()
    cur_clean = cur_col.dropna()
    
    if len(ref_clean) < 5 or len(cur_clean) < 5:
        return False, 1.0
    
    statistic, p_value = stats.ks_2samp(ref_clean, cur_clean)
    return p_value < threshold, p_value


def chi2_test_drift(ref_col: pd.Series, cur_col: pd.Series, threshold: float = 0.05) -> Tuple[bool, float]:
    """Chi-squared test for categorical drift."""
    from scipy import stats
    
    all_categories = set(ref_col.dropna().unique()) | set(cur_col.dropna().unique())
    
    ref_counts = ref_col.value_counts()
    cur_counts = cur_col.value_counts()
    
    ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
    cur_freq = [cur_counts.get(cat, 0) for cat in all_categories]
    
    # Normalize to same scale
    ref_sum = sum(ref_freq) or 1
    cur_sum = sum(cur_freq) or 1
    ref_freq = [f / ref_sum * 100 for f in ref_freq]
    cur_freq = [f / cur_sum * 100 for f in cur_freq]
    
    # Add small constant to avoid zeros
    ref_freq = [f + 0.01 for f in ref_freq]
    cur_freq = [f + 0.01 for f in cur_freq]
    
    try:
        statistic, p_value = stats.chisquare(cur_freq, ref_freq)
        return p_value < threshold, p_value
    except:
        return False, 1.0


def run_fallback_drift_detection(
    reference: pd.DataFrame, 
    current: pd.DataFrame
) -> Dict[str, Any]:
    """Run statistical drift detection without Evidently."""
    print("🔬 Running statistical drift analysis (fallback mode)...")
    
    results = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "drifted_count": 0,
        "total_features": 0,
        "drifted_features": [],
        "feature_details": {}
    }
    
    common_cols = list(set(reference.columns) & set(current.columns))
    results["total_features"] = len(common_cols)
    
    for col in common_cols:
        ref_col = reference[col]
        cur_col = current[col]
        
        # Determine if numeric or categorical
        if pd.api.types.is_numeric_dtype(ref_col):
            drift_detected, p_value = ks_test_drift(ref_col, cur_col)
            test_type = "KS-test"
        else:
            drift_detected, p_value = chi2_test_drift(ref_col, cur_col)
            test_type = "Chi2-test"
        
        results["feature_details"][col] = {
            "drift_detected": drift_detected,
            "p_value": p_value,
            "test": test_type
        }
        
        if drift_detected:
            results["drifted_count"] += 1
            results["drifted_features"].append(col)
    
    results["drift_share"] = results["drifted_count"] / max(results["total_features"], 1)
    results["dataset_drift"] = results["drift_share"] > 0.3
    
    return results


# =============================================================================
# MAIN: Drift Detection with Evidently or Fallback
# =============================================================================

def run_drift_detection() -> Dict[str, Any]:
    """
    Run drift detection comparing reference to current data.
    Uses Evidently if available, otherwise falls back to statistical tests.
    """
    print("=" * 70)
    print("🔍 DRIFT DETECTION ANALYSIS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Mode: {'Evidently' if EVIDENTLY_AVAILABLE else 'Statistical Fallback'}")
    print()
    
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
    
    # Run drift detection
    if EVIDENTLY_AVAILABLE:
        results = _run_evidently_drift(ref_aligned, cur_aligned)
    else:
        results = run_fallback_drift_detection(ref_aligned, cur_aligned)
    
    # Save results
    Path(REPORT_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📋 Saved JSON report: {REPORT_JSON_PATH}")
    
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


def _run_evidently_drift(ref_aligned: pd.DataFrame, cur_aligned: pd.DataFrame) -> Dict[str, Any]:
    """Run Evidently-based drift detection."""
    print("🔬 Running Evidently drift analysis...")
    
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable()
    ])
    
    report.run(
        reference_data=ref_aligned,
        current_data=cur_aligned
    )
    
    # Save HTML report
    report.save_html(REPORT_HTML_PATH)
    print(f"📊 Saved HTML report: {REPORT_HTML_PATH}")
    
    # Parse results
    report_dict = report.as_dict()
    return _parse_evidently_results(report_dict)


def _parse_evidently_results(report_dict: Dict) -> Dict[str, Any]:
    """Parse Evidently report dictionary."""
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
    """Check if drift exceeds threshold."""
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
        import traceback
        traceback.print_exc()
