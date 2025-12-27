"""
pipelines/local_pipeline.py
============================
PRODUCTION PIPELINE - Clinical Trial Dropout Prediction

Single-command execution of complete MLOps workflow:
Data → Features → Preprocessing → Training → MLflow Tracking

Usage:
    python pipelines/local_pipeline.py

Expected Runtime: ~3 minutes
Expected Output: ROC-AUC 0.64 ± 0.02
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.preprocess import preprocess_data
from core.train import train_model

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    target: str = "dropout",
    feature_version: str = "v3_causal",
    model_type: str = "xgboost"
) -> Dict[str, Any]:
    """
    Execute end-to-end production pipeline.
    
    Pipeline Steps:
    1. Feature Engineering (rates, interactions, domain)
    2. Preprocessing (scaling, versioning)
    3. Model Training (class balancing, CV)
    4. MLflow Logging (parameters, metrics, artifacts)
    
    Args:
        target: Target variable ('dropout', 'early_dropout', etc.)
        feature_version: Feature set identifier ('v3_causal')
        model_type: Model selection ('logistic', 'xgboost', 'lightgbm')
        
    Returns:
        dict: {'model': trained_model, 'metrics': performance_dict}
    """
    start_time = time.time()
    
    # Pipeline header
    logger.info("=" * 80)
    logger.info("🚀 CLINICAL TRIAL DROPOUT PREDICTION - PRODUCTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Target:          {target}")
    logger.info(f"Feature Version: {feature_version}")
    logger.info(f"Model Type:      {model_type}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Feature Engineering & Preprocessing
        logger.info("\n📊 STEP 1: Feature Engineering")
        logger.info("-" * 80)
        step1_start = time.time()
        
        preprocess_data(
            input_path="data/raw/clinical_trials.csv",
            output_path=f"data/processed/clinical_trials_{target}.csv",
            target=target,
            feature_version=feature_version
        )
        
        step1_time = time.time() - step1_start
        logger.info(f"✅ Preprocessing complete ({step1_time:.1f}s)")
        
        # Step 2: Model Training
        logger.info("\n🎯 STEP 2: Model Training")
        logger.info("-" * 80)
        step2_start = time.time()
        
        result = train_model(
            data_path=f"data/processed/clinical_trials_{target}.csv",
            target=target,
            feature_version=feature_version,
            model_type=model_type
        )
        
        step2_time = time.time() - step2_start
        logger.info(f"✅ Training complete ({step2_time:.1f}s)")
        
        # Summary
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Model:         {model_type.upper()}")
        logger.info(f"CV ROC-AUC:    {result['metrics']['cv_roc_auc']:.3f} ± {result['metrics']['cv_roc_auc_std']:.3f}")
        logger.info(f"Test ROC-AUC:  {result['metrics']['test_roc_auc']:.3f}")
        logger.info(f"Recall:        {result['metrics']['recall']:.3f}")
        logger.info(f"Precision:     {result['metrics']['precision']:.3f}")
        logger.info(f"F1 Score:      {result['metrics']['f1_score']:.3f}")
        logger.info(f"\n⏱️  Total Runtime: {total_time:.1f}s")
        logger.info("\n💡 View detailed results:")
        logger.info("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
        logger.info("   Open: http://localhost:5000")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {str(e)}")
        logger.error(f"Error Type: {type(e).__name__}")
        raise


def run_comparison() -> None:
    """
    Run all 3 models and compare performance.
    
    Purpose: Demonstrate feature effectiveness across models
    Output: Comparison table sorted by ROC-AUC
    """
    print("\n" + "=" * 80)
    print("🔬 PRODUCTION PIPELINE - MODEL COMPARISON")
    print("=" * 80)
    print("Testing 3 models to validate feature engineering:")
    print("  • Logistic Regression (baseline)")
    print("  • XGBoost (gradient boosting)")
    print("  • LightGBM (fast boosting)")
    print("=" * 80)
    
    results = {}
    total_start = time.time()
    
    for model_type in ["logistic", "xgboost", "lightgbm"]:
        print(f"\n\n{'#' * 80}")
        print(f"# MODEL: {model_type.upper()}")
        print(f"{'#' * 80}\n")
        
        try:
            result = run_pipeline(
                target="dropout",
                feature_version="v3_causal",
                model_type=model_type
            )
            results[model_type] = result['metrics']['test_roc_auc']
            
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            results[model_type] = 0.0
    
    # Final comparison
    total_time = time.time() - total_start
    
    print("\n\n" + "=" * 80)
    print("📊 FINAL COMPARISON (v3_causal features)")
    print("=" * 80)
    
    for model, roc_auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        status = "✅" if roc_auc > 0.60 else "❌"
        print(f"   {status} {model.upper():15s} ROC-AUC: {roc_auc:.3f}")
    
    print("=" * 80)
    print(f"⏱️  Total Pipeline Runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Validation check
    best_roc_auc = max(results.values())
    if best_roc_auc > 0.65:
        print("\n✅ SUCCESS: Learnable signal confirmed (ROC-AUC > 0.65)")
    elif best_roc_auc > 0.60:
        print("\n⚠️  WARNING: Marginal signal (0.60 < ROC-AUC < 0.65)")
    else:
        print("\n❌ FAILURE: No learnable signal (ROC-AUC < 0.60)")
    
    print("=" * 80)


if __name__ == "__main__":
    # Run full comparison (3 models)
    run_comparison()
