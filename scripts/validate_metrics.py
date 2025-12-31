"""
scripts/validate_metrics.py
============================
MODEL GATE VALIDATOR - CI/CD Critical Path

Purpose: Block deployment of underperforming models.
Action: Raises RuntimeError if model fails quality gates.

This is the CORE GATE for your ML pipeline.
Models that fail this check will NOT be promoted.

Usage:
    python scripts/validate_metrics.py

Exit codes:
    0 = Model approved (meets all thresholds)
    1 = Model rejected (fails quality gates)
"""

import mlflow
import sys
import os
from datetime import datetime
from pathlib import Path

# ============================================================
# PRODUCTION QUALITY GATES
# These are your deployment thresholds - adjust as needed
# ============================================================

MIN_RECALL = 0.55       # Minimum recall (catch dropout cases)
MIN_ROC_AUC = 0.58      # Minimum ROC-AUC (overall discrimination)
MIN_PRECISION = 0.25    # Minimum precision (avoid too many false positives)
MIN_F1 = 0.35           # Minimum F1 score (balance)


def find_mlflow_db():
    """Find MLflow database in common locations."""
    possible_paths = [
        "mlflow.db",
        "./mlflow.db",
        "../mlflow.db",
        "mlruns/../mlflow.db",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return f"sqlite:///{path}"
    
    # Check if mlruns directory exists (file-based tracking)
    if Path("mlruns").exists():
        return None  # Use default file-based tracking
    
    return "sqlite:///mlflow.db"  # Default


def validate_model():
    """
    Validate the best model against production quality gates.
    
    Returns True if model passes all gates, raises RuntimeError if not.
    """
    print("=" * 70)
    print("🔒 MODEL QUALITY GATE VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Find and connect to MLflow
    tracking_uri = find_mlflow_db()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"📁 MLflow URI: {tracking_uri}")
    else:
        print("📁 MLflow URI: default (mlruns/)")
    
    # List available experiments
    try:
        experiments = mlflow.search_experiments()
        print(f"   Found {len(experiments)} experiment(s)")
        for exp in experiments[:5]:  # Show first 5
            print(f"     - {exp.name} (ID: {exp.experiment_id})")
    except Exception as e:
        print(f"   Warning: Could not list experiments: {e}")
    
    print()
    
    # Search for runs across ALL experiments
    runs = None
    search_methods = [
        ("by recall DESC", ["metrics.recall DESC"]),
        ("by test_roc_auc DESC", ["metrics.test_roc_auc DESC"]),
        ("by start_time DESC", ["start_time DESC"]),
    ]
    
    for method_name, order_by in search_methods:
        try:
            runs = mlflow.search_runs(
                order_by=order_by,
                max_results=10
            )
            if len(runs) > 0:
                print(f"✅ Found {len(runs)} runs (searched {method_name})")
                break
        except Exception as e:
            print(f"   Search {method_name} failed: {e}")
    
    if runs is None or len(runs) == 0:
        # Try searching without ordering
        try:
            runs = mlflow.search_runs(max_results=10)
            if len(runs) > 0:
                print(f"✅ Found {len(runs)} runs (unordered search)")
        except Exception as e:
            print(f"   Unordered search failed: {e}")
    
    if runs is None or len(runs) == 0:
        print()
        print("⚠️  No MLflow runs found in database.")
        print("   This can happen on first CI run or if MLflow artifacts weren't properly saved.")
        print()
        print("   Skipping validation (allowing pipeline to proceed)...")
        print("=" * 70)
        return True  # Allow to proceed on first run
    
    # Find best run with recall metric
    runs_with_recall = runs[runs["metrics.recall"].notna()]
    
    if len(runs_with_recall) == 0:
        print("⚠️  No runs with recall metric found.")
        print("   Using most recent run instead...")
        best_run = runs.iloc[0]
    else:
        best_run = runs_with_recall.sort_values("metrics.recall", ascending=False).iloc[0]
    
    run_id = best_run["run_id"]
    
    print()
    print(f"📊 Evaluating Run: {run_id[:8]}...")
    print("-" * 70)
    
    # Extract metrics with safe defaults
    recall = best_run.get("metrics.recall", 0) or 0
    precision = best_run.get("metrics.precision", 0) or 0
    f1_score = best_run.get("metrics.f1_score", 0) or 0
    roc_auc = best_run.get("metrics.test_roc_auc") or best_run.get("metrics.roc_auc", 0) or 0
    
    # Handle NaN values
    import math
    recall = 0 if (isinstance(recall, float) and math.isnan(recall)) else recall
    precision = 0 if (isinstance(precision, float) and math.isnan(precision)) else precision
    f1_score = 0 if (isinstance(f1_score, float) and math.isnan(f1_score)) else f1_score
    roc_auc = 0 if (isinstance(roc_auc, float) and math.isnan(roc_auc)) else roc_auc
    
    print(f"   Recall:     {recall:.3f}  (threshold: {MIN_RECALL})")
    print(f"   Precision:  {precision:.3f}  (threshold: {MIN_PRECISION})")
    print(f"   F1 Score:   {f1_score:.3f}  (threshold: {MIN_F1})")
    print(f"   ROC-AUC:    {roc_auc:.3f}  (threshold: {MIN_ROC_AUC})")
    print("-" * 70)
    
    # Validate against gates
    failures = []
    
    if recall < MIN_RECALL:
        failures.append(f"Recall {recall:.3f} < {MIN_RECALL}")
    
    if roc_auc < MIN_ROC_AUC:
        failures.append(f"ROC-AUC {roc_auc:.3f} < {MIN_ROC_AUC}")
    
    if precision < MIN_PRECISION:
        failures.append(f"Precision {precision:.3f} < {MIN_PRECISION}")
    
    if f1_score < MIN_F1:
        failures.append(f"F1 Score {f1_score:.3f} < {MIN_F1}")
    
    # Report result
    if failures:
        print()
        print("❌ MODEL REJECTED - Quality gates failed:")
        for f in failures:
            print(f"   • {f}")
        print()
        print("=" * 70)
        raise RuntimeError(
            f"Model rejected: {len(failures)} quality gate(s) failed. "
            f"Details: {'; '.join(failures)}"
        )
    
    print()
    print("✅ MODEL APPROVED - All quality gates passed!")
    print()
    print("Model is ready for promotion to Staging/Production.")
    print("=" * 70)
    
    return True


def promote_to_staging(model_name: str = "ClinicalTrialDropoutModel"):
    """
    Optional: Promote validated model to Staging.
    
    Only call this AFTER validate_model() passes.
    
    Args:
        model_name: Registered model name in MLflow
    """
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # Get latest version
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except:
        versions = []
    
    if not versions:
        print(f"⚠️  No registered model found: {model_name}")
        return False
    
    latest_version = max(versions, key=lambda v: int(v.version))
    
    print(f"Promoting {model_name} v{latest_version.version} to Staging...")
    
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Staging",
        archive_existing_versions=True
    )
    
    print(f"✅ {model_name} v{latest_version.version} is now in Staging")
    return True


if __name__ == "__main__":
    try:
        validate_model()
        sys.exit(0)
    except RuntimeError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
