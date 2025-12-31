"""
src/core/train.py
==================
MODEL COMPARISON & EVALUATION MODULE

Purpose:
- Compare multiple ML algorithms (LightGBM, XGBoost, Logistic Regression)
- Log experiments to MLflow for model selection
- DO NOT use for production deployment (use MLflow registry instead)

Golden Path: Step 4 - Model training and comparison
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import logging
import subprocess
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("clinical_trial_dropout_causal_signal")


def get_dvc_data_version() -> str:
    """
    Get DVC data version hash for reproducibility tracking.
    
    Links model runs to exact data snapshots via DVC.
    Critical for data-model lineage in production MLOps.
    
    Returns:
        str: DVC status JSON or 'no-dvc' if DVC not available
    """
    try:
        result = subprocess.run(
            ["dvc", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip() if result.stdout else "dvc-clean"
        return "dvc-unavailable"
    except Exception as e:
        logger.warning(f"Could not get DVC version: {e}")
        return "no-dvc"



def train_model(
    data_path: str,
    target: str = "dropout",
    feature_version: str = "v3_causal",
    model_type: str = "xgboost"
) -> dict:
    """
    Train production model with CAUSAL features.
    
    Args:
        data_path: Path to processed CSV
        target: Target column name
        feature_version: Feature set identifier
        model_type: 'logistic', 'xgboost', or 'lightgbm'
        
    Returns:
        Dict with model and metrics
    """
    # Load processed data
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]
    
    logger.info(f"📊 Data: {X.shape}")
    logger.info(f"📊 Target distribution: {y.value_counts(normalize=True).to_dict()}")
    
    # Check class imbalance
    imbalance_ratio = y.value_counts(normalize=True).min()
    if imbalance_ratio < 0.3:
        logger.warning(f"⚠️ Class imbalance detected: {imbalance_ratio:.1%} minority class")
    
    # Stratified split (CRITICAL for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"✅ Stratified split: train={y_train.value_counts(normalize=True).to_dict()}")
    
    # Calculate class weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    with mlflow.start_run(run_name=f"{model_type}_{target}_{feature_version}"):
        
        # Select model
        if model_type == "logistic":
            logger.info("🎯 Training Logistic Regression with class_weight='balanced'")
            
            # IMPORTANT: Use pipeline with StandardScaler
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    class_weight='balanced',  # Handle imbalance
                    max_iter=1000,
                    random_state=42
                ))
            ])
            
        elif model_type == "xgboost":
            logger.info("🎯 Training XGBoost with scale_pos_weight")
            
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,  # Handle imbalance
                random_state=42,
                eval_metric='logloss'
            )
            
        elif model_type == "lightgbm":
            logger.info("🎯 Training LightGBM with FAST & SAFE config (prevents hanging)")
            
            # FAST & SAFE TEMPLATE - prevents hanging on Windows
            model = lgb.LGBMClassifier(
                # Boosting config
                boosting_type="gbdt",
                learning_rate=0.05,        # Raised (was 0.1)
                n_estimators=500,          # Reduced (was 200, could hang at 5000)
                
                # Tree complexity caps (CRITICAL - prevents hanging)
                num_leaves=31,
                max_depth=10,              # Increased from 5
                min_data_in_leaf=50,       # NEW: prevents overfitting + speeds up
                max_bin=128,               # NEW: histogram optimization
                
                # Threading control (prevents deadlocks)
                n_jobs=4,                  # Locked to 4 threads
                device_type="cpu",         # Force CPU (no GPU ambiguity)
                
                # Class imbalance
                class_weight='balanced',
                
                # Monitoring
                verbose=1,                 # Show progress (was -1)
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Cross-validation on training set
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(5),
            scoring='roc_auc'
        )
        cv_roc_auc = cv_scores.mean()
        
        # Test set evaluation
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "cv_roc_auc": cv_roc_auc,
            "cv_roc_auc_std": cv_scores.std(),
            "test_roc_auc": roc_auc_score(y_test, y_pred_proba),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        
        # Log to MLflow (including DVC data version for reproducibility)
        dvc_version = get_dvc_data_version()
        
        mlflow.log_params({
            "target": target,
            "feature_version": feature_version,
            "model_type": model_type,
            "scale_pos_weight": scale_pos_weight if model_type == "xgboost" else "balanced",
            "n_features": X.shape[1],
            "data_version": dvc_version  # Links model to exact data snapshot
        })
        
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"ClinicalTrialDropout_{target}_{feature_version}"
        )
        
        # Save model locally for API use
        import joblib
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{model_type}_fixed.pkl"
        model_path = models_dir / model_filename
        joblib.dump(model, model_path)
        logger.info(f"💾 Model saved to {model_path}")
        
        logger.info(f"✅ Model trained")
        logger.info(f"   CV ROC-AUC:   {metrics['cv_roc_auc']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
        logger.info(f"   Test ROC-AUC: {metrics['test_roc_auc']:.3f}")
        logger.info(f"   Recall:       {metrics['recall']:.3f}")
        
        return {"model": model, "metrics": metrics}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test all model types
    for model_type in ["logistic", "xgboost", "lightgbm"]:
        print(f"\n{'='*80}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*80}")
        
        result = train_model(
            data_path="data/processed/clinical_trials_processed.csv",
            target="dropout",
            feature_version="v3_causal",
            model_type=model_type
        )
        
        print(f"\n✅ {model_type.upper()} Results:")
        print(f"   Test ROC-AUC: {result['metrics']['test_roc_auc']:.3f}")
