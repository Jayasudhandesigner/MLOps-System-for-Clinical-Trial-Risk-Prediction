"""
diagnose_model.py
==================
Diagnose why the model is predicting incorrectly
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load raw and processed data
raw = pd.read_csv('data/raw/clinical_trials_realistic_v5.csv')
processed = pd.read_csv('data/processed/clinical_trials_dropout.csv')

print("="*80)
print("MODEL DIAGNOSIS")
print("="*80)

# Check raw data signal
print("\n1. RAW DATA SIGNAL STRENGTH")
print("-"*80)

low_risk = raw[(raw['age'] < 40) & (raw['treatment_group']=='Active')]
high_risk = raw[(raw['age'] > 65) & (raw['treatment_group']=='Placebo')]

print(f"Low-risk patients dropout rate: {low_risk['dropout'].mean():.2%}")
print(f"High-risk patients dropout rate: {high_risk['dropout'].mean():.2%}")
print(f"Signal strength (difference): {abs(high_risk['dropout'].mean() - low_risk['dropout'].mean()):.2%}")

# Check processed features
print("\n2. PROCESSED FEATURES")
print("-"*80)
print(f"Features: {[col for col in processed.columns if col != 'dropout']}")
print(f"\nFeature stats:")
for col in processed.columns:
    if col != 'dropout':
        print(f"  {col:25s}: mean={processed[col].mean():7.3f}, std={processed[col].std():7.3f}")

# Check correlation
print("\n3. FEATURE-TARGET CORRELATION")
print("-"*80)
correlations = processed.corr()['dropout'].drop('dropout').sort_values(key=abs, ascending=False)
print(correlations)

# Identify issue
print("\n4. DIAGNOSIS")
print("-"*80)

if abs(high_risk['dropout'].mean() - low_risk['dropout'].mean()) < 0.3:
    print("⚠️  WEAK SIGNAL: Data doesn't have strong enough separation")
    print("   Solution: Regenerate data with stronger risk factors")
elif correlations.abs().max() < 0.2:
    print("⚠️  FEATURE ISSUE: Engineered features don't correlate with target")
    print("   Solution: Fix feature engineering")
else:
    print("✅ Data looks good. Issue might be:")
    print("   1. Model hyperparameters")
    print("   2. Scaling destroying signal")
    print("   3. Need more training data")

# Test a simple prediction manually
print("\n5. MANUAL PREDICTION TEST")
print("-"*80)

# Load preprocessor and model
import joblib
import mlflow

try:
    preprocessor = joblib.load('data/processed/preprocessor_dropout_v3_causal.pkl')
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model("models:/ClinicalTrialDropout_dropout_v3_causal/latest")
    
    # Create a very low risk patient 
    low_risk_patient = pd.DataFrame([{
        'age': 30,
        'days_in_trial': 60,
        'visit_rate': 1.0,  # Perfect compliance
        'adverse_event_rate': 0.0,  # No events
        'time_since_last_visit': 2,  # Recent visit
        'burden': 0.0,
        'age_adverse_risk': 0.0,
        'trial_phase_risk': 0.2,  # Phase I
        'treatment_risk': 0.1  # Active
    }])
    
    # Create a very high risk patient
    high_risk_patient = pd.DataFrame([{
        'age': 75,
        'days_in_trial': 180,
        'visit_rate': 0.1,  # Very poor compliance
        'adverse_event_rate': 0.05,  # Many events
        'time_since_last_visit': 120,  # Long time
        'burden': 0.045,
        'age_adverse_risk': 0.044,
        'trial_phase_risk': 0.8,  # Phase III
        'treatment_risk': 0.4  # Placebo
    }])
    
    # Scale and predict
    low_scaled = preprocessor.transform(low_risk_patient)
    high_scaled = preprocessor.transform(high_risk_patient)
    
    low_prob = model.predict_proba(low_scaled)[0, 1]
    high_prob = model.predict_proba(high_scaled)[0, 1]
    
    print(f"Low-risk patient prediction: {low_prob:.3f} (should be ~0.1-0.3)")
    print(f"High-risk patient prediction: {high_prob:.3f} (should be ~0.7-0.9)")
    
    if low_prob > high_prob:
        print("\n🚨 INVERTED PREDICTIONS CONFIRMED")
        print("   Model is predicting OPPOSITE of expected")
    elif abs(low_prob - high_prob) < 0.2:
        print("\n⚠️  MODEL NOT DISCRIMINATING")
        print("   Model predictions are too similar")
    else:
        print("\n✅ Model predictions look correct")
        
except Exception as e:
    print(f"Error loading model: {e}")

print("\n" + "="*80)
