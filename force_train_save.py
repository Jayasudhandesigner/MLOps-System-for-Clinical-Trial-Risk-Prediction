"""
force_train_save.py
===================
Train and save the model unconditionally.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import os
import sys

print("Starting training...")

# Load raw data
try:
    df = pd.read_csv('data/raw/clinical_trials_realistic_v5.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Create features matching the "Simple Config"
features = pd.DataFrame()
features['age'] = df['age']
features['days_in_trial'] = df['days_in_trial']
expected_visits = df['days_in_trial'] / 30
features['visits_completed'] = df['visits_completed']
features['visit_compliance'] = df['visits_completed'] / (expected_visits + 0.1)
features['time_since_last_visit'] = df['days_in_trial'] - df['last_visit_day']
features['adverse_events'] = df['adverse_events']
features['adverse_rate'] = df['adverse_events'] / (df['days_in_trial'] + 1)
phase_map = {'Phase I': 1, 'Phase II': 2, 'Phase III': 3}
treatment_map = {'Active': 1, 'Control': 2, 'Placebo': 3}
features['trial_phase_encoded'] = df['trial_phase'].map(phase_map)
features['treatment_encoded'] = df['treatment_group'].map(treatment_map)

y = df['dropout']

# Train
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42
)

model.fit(features, y)
print("Model trained.")

# Save
if not os.path.exists('models'):
    os.makedirs('models')
    
joblib.dump(model, 'models/xgboost_fixed.pkl')
print("Model saved to models/xgboost_fixed.pkl")

# Also save the dummy preprocessor because the API expects one
# The API in main.py loads a preprocessor.
# We need to make sure the API uses OUR feature engineering logic, 
# not the old one.
# So we need to update the API code to do feature engineering properly 
# if we change the model features.

print("Done.")
