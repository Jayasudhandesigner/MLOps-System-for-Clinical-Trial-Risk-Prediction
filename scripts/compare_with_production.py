"""
scripts/compare_with_production.py
===================================
MODEL PROMOTION GATE - Prevents regression

Compares the newly trained 'candidate' model against the current 'Production' model.
Only promotes if the candidate performs better or equal to production.

Usage:
    python scripts/compare_with_production.py

Exits with:
    0 - Candidate approved (better/equal or no production model exists)
    1 - Candidate rejected (worse performance)
"""

import mlflow
import sys
import logging
import os
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "ClinicalTrialDropoutModel"
METRIC_TO_COMPARE = "recall"  # Primary metric from user requirements

def main():
    logger.info("=" * 60)
    logger.info("🛡️  MODEL PROMOTION GATE")
    logger.info("=" * 60)

    # Use persistent DB
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    # 1. Get Candidate Model (latest in 'None' stage)
    try:
        candidates = client.get_latest_versions(MODEL_NAME, stages=["None"])
        if not candidates:
            logger.error("❌ No candidate models found in 'None' stage.")
            sys.exit(1)
        candidate = candidates[0]
        logger.info(f"🆕 Candidate Model: v{candidate.version}")
    except Exception as e:
        logger.error(f"❌ Failed to get candidate model: {e}")
        sys.exit(1)

    # 2. Get Production Model
    try:
        production_models = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not production_models:
            logger.info("⚠️  No Production model found. Auto-approving candidate.")
            promote_model(client, candidate)
            return
        
        production = production_models[0]
        logger.info(f"🏭 Production Model: v{production.version}")
        
    except Exception as e:
        logger.warning(f"⚠️  Error fetching production model: {e}")
        logger.info("Assuming first deployment -> Auto-approving.")
        promote_model(client, candidate)
        return

    # 3. Compare Metrics
    try:
        # Fetch runs to get metrics
        cand_run = client.get_run(candidate.run_id)
        prod_run = client.get_run(production.run_id)

        cand_metric = cand_run.data.metrics.get(METRIC_TO_COMPARE, 0.0)
        prod_metric = prod_run.data.metrics.get(METRIC_TO_COMPARE, 0.0)

        logger.info("-" * 40)
        logger.info(f"📊 COMPARISON ({METRIC_TO_COMPARE})")
        logger.info(f"   Candidate (v{candidate.version}): {cand_metric:.3f}")
        logger.info(f"   Production (v{production.version}): {prod_metric:.3f}")
        logger.info("-" * 40)

        # Logic: Candidate must be at least as good as production
        # Allows equal performance (updates are good for data freshness)
        if cand_metric >= prod_metric:
            logger.info("✅ Candidate approved (Better or Equal Performance)")
            promote_model(client, candidate)
        else:
            logger.error(f"❌ Candidate rejected (Worse {METRIC_TO_COMPARE})")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        sys.exit(1)

def promote_model(client, model_version):
    """Promote model to Production stage."""
    try:
        logger.info(f"🚀 Promoting v{model_version.version} to Production...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info("✅ Promotion Successful!")
    except Exception as e:
        logger.error(f"❌ Promotion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
