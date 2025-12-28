"""
fix_model.py
============
Retrain model with better configuration to fix inverted predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import joblib
import mlflow
import mlflow.sklearn

print("\n" + "="*80)
print("FIXING MODEL - RETRAINING WITH CORRECTED APPROACH")
print("="*80 + "\n")

# Load raw data
df = pd.read_csv('data/raw/clinical_trials_realistic_v5.csv')

print(f"Loaded {len(df)} patients")
print(f"Dropout rate: {df['dropout'].mean():.2%}\n")

# Create SIMPLE, DIRECT features (no complex engineering)
print("Creating simple, interpretable features...")

features = pd.DataFrame()

# Direct features (already in correct direction)
features['age'] = df['age']  # Higher = more risk
features['days_in_trial'] = df['days_in_trial']  # Longer = more dropout chance

# Visit compliance (INVERTED - more visits = less risk)
expected_visits = df['days_in_trial'] / 30
features['visits_completed'] = df['visits_completed']  # More = LESS risk
features['visit_compliance'] = df['visits_completed'] / (expected_visits + 0.1)  # Higher = LESS risk

# Time since last visit (more = worse)
features['time_since_last_visit'] = df['days_in_trial'] - df['last_visit_day']  # Higher = more risk

# Adverse events (more = worse)
features['adverse_events'] = df['adverse_events']  # Higher = more risk
features['adverse_rate'] = df['adverse_events'] / (df['days_in_trial'] + 1)  # Higher = more risk

# Encode categoricals as RISK scores
phase_map = {'Phase I': 1, 'Phase II': 2, 'Phase III': 3}
treatment_map = {'Active': 1, 'Control': 2, 'Placebo': 3}

features['trial_phase_encoded'] = df['trial_phase'].map(phase_map)  # Higher = more risk
features['treatment_encoded'] = df['treatment_group'].map(treatment_map)  # Higher = more risk

# Target
y = df['dropout']

print(f"Created {len(features.columns)} features")
print(f"Features: {list(features.columns)}\n")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

# Train simple XGBoost
print("Training XGBoost with simple configuration...\n")

model = xgb.XGBClassifier(
    n_estimators=100,  # Reduced
    max_depth=4,       # Shallow trees
    learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"✅ Training complete!")
print(f"\nTest ROC-AUC: {roc_auc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Dropout']))

# Feature importance
print("\nFeature Importance:")
importances = pd.DataFrame({
    'feature': features.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importances)

# Manual test
print("\n" + "="*80)
print("MANUAL PREDICTION TEST")
print("="*80 + "\n")

# Very low risk patient
low_risk_test = pd.DataFrame([{
    'age': 30,
    'days_in_trial': 60,
    'visits_completed': 3,
    'visit_compliance': 1.5,  # Perfect
    'time_since_last_visit': 2,  # Recent
    'adverse_events': 0,
    'adverse_rate': 0,
    'trial_phase_encoded': 1,  # Phase I
    'treatment_encoded': 1  # Active
}])

# Very high risk patient
high_risk_test = pd.DataFrame([{
    'age': 75,
    'days_in_trial': 180,
    'visits_completed': 1,
    'visit_compliance': 0.17,  # Very poor
    'time_since_last_visit': 120,  # Long time
    'adverse_events': 10,
    'adverse_rate': 0.056,
    'trial_phase_encoded': 3,  # Phase III
    'treatment_encoded': 3  # Placebo
}])

low_prob = model.predict_proba(low_risk_test)[0, 1]
high_prob = model.predict_proba(high_risk_test)[0, 1]

print(f"Low-risk patient dropout prob:  {low_prob:.3f} (expect < 0.3)")
print(f"High-risk patient dropout prob: {high_prob:.3f} (expect > 0.7)")

if low_prob < high_prob and roc_auc > 0.65:
    print("\n✅ MODEL IS WORKING CORRECTLY!")
    print("   Saving model...")
    
    # Save model with MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("clinical_trial_dropout_fixed")
    
    with mlflow.start_run(run_name="xgboost_fixed"):
        mlflow.log_params({
            "model_type": "xgboost",
            "n_estimators": 100,
            "max_depth": 4,
            "feature_engineering": "simple_direct"
        })
        mlflow.log_metrics({
            "test_roc_auc": roc_auc,
            "low_risk_prob": low_prob,
            "high_risk_prob": high_prob
        })
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="ClinicalTrialDropout_dropout_v3_causal"
        )
    
    # Also save as pickle for direct loading
    joblib.dump(model, 'models/xgboost_fixed.pkl')
    print("   ✅ Model saved to models/xgboost_fixed.pkl")
    print("   ✅ Model registered in MLflow")
    
else:
    print("\n⚠️  MODEL STILL HAS ISSUES")
    print(f"   ROC-AUC: {roc_auc:.3f} (need > 0.65)")
    print(f"   Low risk prob: {low_prob:.3f}")
    print(f"   High risk prob: {high_prob:.3f}")

print("\n" + "="*80)
