"""
src/core/train.py
==================
PRODUCTION TRAINING MODULE (CAUSAL SIGNAL VERSION)

Golden Path: Step 4
Trains production model with ALL optimizations for learnable signal.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import logging
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
            logger.info("🎯 Training LightGBM with class_weight='balanced'")
            
            model = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                max_depth=5,
                learning_rate=0.1,
                class_weight='balanced',  # Handle imbalance
                random_state=42,
                verbose=-1
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
        
        # Log to MLflow
        mlflow.log_params({
            "target": target,
            "feature_version": feature_version,
            "model_type": model_type,
            "scale_pos_weight": scale_pos_weight if model_type == "xgboost" else "balanced",
            "n_features": X.shape[1]
        })
        
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"ClinicalTrialDropout_{target}_{feature_version}"
        )
        
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
