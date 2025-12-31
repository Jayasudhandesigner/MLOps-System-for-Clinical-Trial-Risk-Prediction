"""
pipelines/local_pipeline.py
============================
PRODUCTION PIPELINE - Clinical Trial Dropout Prediction

Single-command execution of complete MLOps workflow:
Data → Features → Preprocessing → Training → MLflow Tracking

Usage:
    python pipelines/local_pipeline.py

Uses Logistic Regression (best performer with ROC-AUC: 0.661)
"""

import sys
import logging
import time
import joblib
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

# Production model configuration
PRODUCTION_MODEL = "logistic"  # Best performer


def run_pipeline(
    target: str = "dropout",
    feature_version: str = "v3_causal",
    model_type: str = PRODUCTION_MODEL
) -> Dict[str, Any]:
    """
    Execute end-to-end production pipeline.
    
    Pipeline Steps:
    1. Feature Engineering (rates, interactions, domain)
    2. Preprocessing (scaling, versioning)
    3. Model Training (class balancing, CV)
    4. MLflow Logging (parameters, metrics, artifacts)
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
        
        preprocess_data(
            input_path="data/raw/clinical_trials_realistic_v5.csv",
            output_path=f"data/processed/clinical_trials_{target}.csv",
            target=target,
            feature_version=feature_version
        )
        
        logger.info("✅ Feature engineering complete")
        
        # Step 2: Model Training
        logger.info("\n🎯 STEP 2: Model Training")
        logger.info("-" * 80)
        
        result = train_model(
            data_path=f"data/processed/clinical_trials_{target}.csv",
            target=target,
            feature_version=feature_version,
            model_type=model_type
        )
        
        # Step 3: Save production model with standard name
        logger.info("\n💾 STEP 3: Saving Production Model")
        logger.info("-" * 80)
        
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as production model (standard name for API)
        production_model_path = models_dir / "production_model.pkl"
        joblib.dump(result['model'], production_model_path)
        logger.info(f"💾 Saved: {production_model_path}")
        
        # Also save with model type name for backwards compatibility
        model_path = models_dir / f"{model_type}_fixed.pkl"
        joblib.dump(result['model'], model_path)
        logger.info(f"💾 Saved: {model_path}")
        
        # Report
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("📈 PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"   Model:      {model_type}")
        logger.info(f"   ROC-AUC:    {result['metrics']['test_roc_auc']:.3f}")
        logger.info(f"   Recall:     {result['metrics']['recall']:.3f}")
        logger.info(f"   Precision:  {result['metrics']['precision']:.3f}")
        logger.info(f"   Runtime:    {elapsed:.1f}s")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🔬 PRODUCTION PIPELINE - LOGISTIC REGRESSION (BEST MODEL)")
    print("=" * 80)
    print("Training production model with ROC-AUC ~0.66")
    print("=" * 80)
    
    total_start = time.time()
    
    result = run_pipeline(
        target="dropout",
        feature_version="v3_causal",
        model_type=PRODUCTION_MODEL
    )
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("📊 FINAL RESULT")
    print("=" * 80)
    print(f"   ✅ LOGISTIC REGRESSION  ROC-AUC: {result['metrics']['test_roc_auc']:.3f}")
    print("=" * 80)
    print(f"⏱️  Total Runtime: {total_time:.1f}s")
    
    if result['metrics']['test_roc_auc'] > 0.60:
        print("\n✅ SUCCESS: Model ready for production")
    else:
        print("\n⚠️  WARNING: Model performance below threshold")
    
    print("=" * 80)
