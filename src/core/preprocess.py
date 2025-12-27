"""
src/core/preprocess.py
=======================
PRODUCTION PREPROCESSING MODULE (CAUSAL SIGNAL VERSION)

Golden Path: Step 3
Orchestrates feature engineering, scaling, and data preparation.
"""

import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from core.ingest import load_data
from core.features import engineer_rate_features, engineer_interaction_features, engineer_domain_features

logger = logging.getLogger(__name__)

# Feature definitions (RATES not COUNTS)
NUMERIC_FEATURES = [
    "age", "days_in_trial"
]

RATE_FEATURES = [
    "visit_rate",                # Compliance: visits / expected
    "adverse_event_rate",        # Risk: events / days
    "time_since_last_visit"      # Engagement gap
]

INTERACTION_FEATURES = [
    "burden",                    # adverse_rate × (1 - visit_rate)
    "age_adverse_risk"          # (age/85) × adverse_rate
]

DOMAIN_FEATURES = [
    "trial_phase_risk",          # Ordinal: Phase I < II < III
    "treatment_risk"             # Ordinal: Active < Control < Placebo
]

CATEGORICAL_FEATURES = [
    "gender"  # Only keep gender as categorical
]

ALL_NUMERIC = NUMERIC_FEATURES + RATE_FEATURES + INTERACTION_FEATURES + DOMAIN_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline.
    
    Pipeline:
    1. Numeric: Impute → Scale (CRITICAL for linear models)
    2. Categorical: Impute → Drop (one-hot if needed)
    
    Returns:
        Fitted ColumnTransformer
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())  # Mean=0, Std=1
    ])
    
    # For simplicity, drop categorical (risk scores encode the important info)
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, ALL_NUMERIC),
    ], remainder='drop')
    
    return preprocessor


def preprocess_data(
    input_path: str,
    output_path: str,
    target: str = "dropout",
    feature_version: str = "v3_causal"
) -> pd.DataFrame:
    """
    Main preprocessing function.
    
    Args:
        input_path: Path to raw CSV
        output_path: Path to save processed CSV
        target: Target variable name
        feature_version: Feature set identifier for MLflow
        
    Returns:
        Processed DataFrame
    """
    # Load
    df = load_data(input_path)
    logger.info(f"📊 Loaded: {df.shape}")
    
    # Engineer features (RATES + INTERACTIONS + DOMAIN)
    df = engineer_rate_features(df)
    df = engineer_interaction_features(df)
    df = engineer_domain_features(df)
    
    logger.info(f"✅ Feature engineering complete (version: {feature_version})")
    
    # Select features
    X = df[ALL_NUMERIC + CATEGORICAL_FEATURES]
    y = df[target]
    
    # Build and fit preprocessor
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Save preprocessor
    preprocessor_path = Path(output_path).parent / f"preprocessor_{target}_{feature_version}.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"💾 Saved preprocessor: {preprocessor_path}")
    
    # Create output DataFrame
    feature_names = ALL_NUMERIC
    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    processed_df[target] = y.values
    
    # Save
    processed_df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved processed data: {output_path} {processed_df.shape}")
    logger.info(f"📈 Target distribution: {y.value_counts(normalize=True).to_dict()}")
    
    return processed_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    preprocess_data(
        input_path="data/raw/clinical_trials.csv",
        output_path="data/processed/clinical_trials_processed.csv",
        target="dropout",
        feature_version="v3_causal"
    )
