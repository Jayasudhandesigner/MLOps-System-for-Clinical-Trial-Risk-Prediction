"""
src/experiments/threshold_tuning.py
====================================
THRESHOLD OPTIMIZATION FOR RECALL MAXIMIZATION

Evaluates multiple decision thresholds (0.2 to 0.6) to find optimal
threshold for maximizing recall while maintaining acceptable precision.

Each threshold creates a separate MLflow run with logged metrics.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import lightgbm as lgb

logger = logging.getLogger(__name__)

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("threshold_tuning_lightgbm")


def evaluate_thresholds(y_true, y_proba, model_name="LightGBM"):
    """
    Evaluate different decision thresholds for dropout prediction.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model being evaluated
        
    Returns:
        List of dicts with threshold results
    """
    thresholds = np.arange(0.2, 0.61, 0.05)  # 0.20, 0.25, ..., 0.60
    results = []

    logger.info(f"🎯 Evaluating {len(thresholds)} thresholds: {thresholds}")

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        result = {
            "threshold": round(threshold, 2),
            "recall": recall,
            "precision": precision,
            "f1": f1
        }
        results.append(result)
        
        # Log each threshold as a separate MLflow run
        with mlflow.start_run(run_name=f"{model_name}_threshold_{threshold:.2f}"):
            mlflow.log_param("decision_threshold", round(threshold, 2))
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1", f1)
            
            logger.info(f"  Threshold {threshold:.2f}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}")

    return results


def train_and_tune_threshold(
    data_path: str,
    target: str = "dropout",
    feature_version: str = "v3_causal"
):
    """
    Train LightGBM model and evaluate multiple thresholds.
    
    Args:
        data_path: Path to processed CSV
        target: Target column name
        feature_version: Feature set identifier
        
    Returns:
        Dict with model, results, and best threshold
    """
    # Load processed data
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]
    
    logger.info(f"📊 Data shape: {X.shape}")
    logger.info(f"📊 Dropout rate: {y.mean():.1%}")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train LightGBM (best model from baseline)
    logger.info("🏗️ Training LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        max_depth=5,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Baseline metrics (default threshold 0.5)
    y_pred_baseline = (y_pred_proba >= 0.5).astype(int)
    baseline_metrics = {
        "recall": recall_score(y_test, y_pred_baseline),
        "precision": precision_score(y_test, y_pred_baseline),
        "f1": f1_score(y_test, y_pred_baseline),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info(f"\n📊 Baseline (threshold=0.5):")
    logger.info(f"   Recall:    {baseline_metrics['recall']:.3f}")
    logger.info(f"   Precision: {baseline_metrics['precision']:.3f}")
    logger.info(f"   F1:        {baseline_metrics['f1']:.3f}")
    logger.info(f"   ROC-AUC:   {baseline_metrics['roc_auc']:.3f}")
    
    # Evaluate thresholds
    logger.info(f"\n🔍 Threshold Tuning:")
    results = evaluate_thresholds(y_test, y_pred_proba, "LightGBM")
    
    # Find best threshold for recall
    best_recall = max(results, key=lambda x: x['recall'])
    best_f1 = max(results, key=lambda x: x['f1'])
    
    logger.info(f"\n🏆 Best Results:")
    logger.info(f"   Best Recall:    threshold={best_recall['threshold']}, recall={best_recall['recall']:.3f}")
    logger.info(f"   Best F1:        threshold={best_f1['threshold']}, f1={best_f1['f1']:.3f}")
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Log parent run with summary
    with mlflow.start_run(run_name="threshold_tuning_summary"):
        mlflow.log_params({
            "model_type": "LightGBM",
            "feature_version": feature_version,
            "n_thresholds_tested": len(results)
        })
        
        mlflow.log_metrics({
            "baseline_recall": baseline_metrics['recall'],
            "baseline_precision": baseline_metrics['precision'],
            "baseline_f1": baseline_metrics['f1'],
            "best_recall": best_recall['recall'],
            "best_recall_threshold": best_recall['threshold'],
            "best_f1": best_f1['f1'],
            "best_f1_threshold": best_f1['threshold']
        })
        
        # Save results as artifact
        results_df.to_csv("threshold_results.csv", index=False)
        mlflow.log_artifact("threshold_results.csv")
        Path("threshold_results.csv").unlink()  # Clean up
    
    return {
        "model": model,
        "results": results_df,
        "baseline_metrics": baseline_metrics,
        "best_recall": best_recall,
        "best_f1": best_f1
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*80)
    logger.info("THRESHOLD TUNING EXPERIMENT - LIGHTGBM")
    logger.info("="*80)
    
    result = train_and_tune_threshold(
        data_path="data/processed/clinical_trials_dropout.csv",
        target="dropout",
        feature_version="v3_causal"
    )
    
    logger.info("\n" + "="*80)
    logger.info("✅ EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"\n📈 Results Summary:")
    print(result['results'].to_string(index=False))
