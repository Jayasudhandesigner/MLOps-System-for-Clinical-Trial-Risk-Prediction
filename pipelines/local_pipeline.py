"""
pipelines/local_pipeline.py
============================
GOLDEN PATH: PRODUCTION PIPELINE (CAUSAL SIGNAL VERSION)

This is the ONLY file recruiters/architects need to run.

Pipeline:
  data/raw (CAUSAL) → ingest → features (RATES) → preprocess → train → MLflow

Usage:
  python pipelines/local_pipeline.py
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.preprocess import preprocess_data
from core.train import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    target: str = "dropout",
    feature_version: str = "v3_causal",
    model_type: str = "xgboost"
):
    """
    Execute end-to-end production pipeline with CAUSAL features.
    
    Args:
        target: Which dropout target to predict
        feature_version: Feature set identifier
        model_type: 'logistic', 'xgboost', or 'lightgbm'
    """
    logger.info("=" * 80)
    logger.info("🚀 CLINICAL TRIAL DROPOUT PREDICTION - CAUSAL SIGNAL PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Target: {target}")
    logger.info(f"Feature Version: {feature_version}")
    logger.info(f"Model: {model_type}")
    logger.info("=" * 80)
    
    # Step 1: Preprocess (RATES + INTERACTIONS + DOMAIN)
    logger.info("\n📊 STEP 1: Feature Engineering (Causal)")
    logger.info("-" * 80)
    preprocess_data(
        input_path="data/raw/clinical_trials.csv",
        output_path=f"data/processed/clinical_trials_{target}.csv",
        target=target,
        feature_version=feature_version
    )
    
    # Step 2: Train (with class balancing)
    logger.info("\n🎯 STEP 2: Model Training (Balanced)")
    logger.info("-" * 80)
    result = train_model(
        data_path=f"data/processed/clinical_trials_{target}.csv",
        target=target,
        feature_version=feature_version,
        model_type=model_type
    )
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model:        {model_type}")
    logger.info(f"CV ROC-AUC:   {result['metrics']['cv_roc_auc']:.3f} ± {result['metrics']['cv_roc_auc_std']:.3f}")
    logger.info(f"Test ROC-AUC: {result['metrics']['test_roc_auc']:.3f}")
    logger.info(f"Recall:       {result['metrics']['recall']:.3f}")
    logger.info(f"F1 Score:     {result['metrics']['f1_score']:.3f}")
    logger.info("\n💡 View results in MLflow:")
    logger.info("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
    logger.info("=" * 80)
    
    return result


if __name__ == "__main__":
    print("\n🔬 TESTING CAUSAL SIGNAL PIPELINE")
    print("=" * 80)
    print("Running 3 models to compare feature effectiveness:")
    print("=" * 80)
    
    results = {}
    
    for model_type in ["logistic", "xgboost", "lightgbm"]:
        print(f"\n\n{'#'*80}")
        print(f"# MODEL: {model_type.upper()}")
        print(f"{'#'*80}\n")
        
        result = run_pipeline(
            target="dropout",
            feature_version="v3_causal",
            model_type=model_type
        )
        
        results[model_type] = result['metrics']['test_roc_auc']
    
    # Final comparison
    print("\n\n" + "=" * 80)
    print("📊 FINAL COMPARISON (v3_causal features)")
    print("=" * 80)
    for model, roc_auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model.upper():15s} ROC-AUC: {roc_auc:.3f}")
    print("=" * 80)
    print("\n✅ If ROC-AUC > 0.65 → Learnable signal confirmed!")
    print("=" * 80)
