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
from datetime import datetime

# ============================================================
# PRODUCTION QUALITY GATES
# These are your deployment thresholds - adjust as needed
# ============================================================

MIN_RECALL = 0.60       # Minimum recall (catch dropout cases)
MIN_ROC_AUC = 0.60      # Minimum ROC-AUC (overall discrimination)
MIN_PRECISION = 0.30    # Minimum precision (avoid too many false positives)
MIN_F1 = 0.40           # Minimum F1 score (balance)


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
    
    # Connect to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Get best run by recall (our primary metric)
    runs = mlflow.search_runs(
        order_by=["metrics.recall DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        raise RuntimeError("No MLflow runs found - cannot validate model")
    
    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    
    print(f"📊 Evaluating Run: {run_id[:8]}...")
    print("-" * 70)
    
    # Extract metrics
    recall = best_run.get("metrics.recall", 0)
    precision = best_run.get("metrics.precision", 0)
    f1_score = best_run.get("metrics.f1_score", 0)
    roc_auc = best_run.get("metrics.test_roc_auc")
    if roc_auc is None:
        roc_auc = best_run.get("metrics.roc_auc", 0)
    
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
    versions = client.search_model_versions(f"name='{model_name}'")
    
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
        sys.exit(1)
