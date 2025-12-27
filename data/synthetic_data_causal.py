"""
CAUSAL SYNTHETIC DATA GENERATION
=================================

Key principle: Dropout is caused by measurable factors, not random chance.

Causal factors:
1. High adverse event rate → Higher dropout risk
2. Low visit compliance → Higher dropout risk  
3. Advanced trial phase → Higher dropout risk
4. Burden (adverse events × poor visits) → Compound risk
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)
n_rows = 1000

print("=" * 80)
print("🔬 GENERATING CAUSAL SYNTHETIC DATA")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Generate base features
# ============================================================================

data = {
    'patient_id': [f'P-{i:04d}' for i in range(1, n_rows + 1)],
    'age': np.random.randint(18, 85, size=n_rows),
    'gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_rows),
    'treatment_group': np.random.choice(['Active', 'Control', 'Placebo'], n_rows),
    'trial_phase': np.random.choice(['Phase I', 'Phase II', 'Phase III'], n_rows),
}

# Temporal features
days_in_trial = np.random.randint(30, 365, size=n_rows)
data['days_in_trial'] = days_in_trial

# ============================================================================
# STEP 2: Generate visits with realistic patterns
# ============================================================================

# Expected: ~1 visit per 30 days
expected_visits = days_in_trial / 30

# Some patients are compliant, others aren't
compliance_factor = np.random.beta(5, 2, n_rows)  # Skewed toward high compliance
visits_completed = np.floor(expected_visits * compliance_factor).astype(int)
visits_completed = np.maximum(visits_completed, 0)  # No negative visits

data['visits_completed'] = visits_completed

# Last visit day (for time_since_last_visit feature)
last_visit_day = np.zeros(n_rows)
for i in range(n_rows):
    if visits_completed[i] > 0:
        # Random day within trial period
        last_visit_day[i] = np.random.randint(
            max(1, days_in_trial[i] - 60),
            days_in_trial[i] + 1
        )
    else:
        last_visit_day[i] = 0

data['last_visit_day'] = last_visit_day.astype(int)

# ============================================================================
# STEP 3: Generate adverse events (correlated with age and trial duration)
# ============================================================================

# Older patients and longer trials → more adverse events
age_normalized = (data['age'] - 18) / 67  # 0 to 1
duration_normalized = days_in_trial / 365  # 0 to 1

# Poisson rate increases with age and duration
adverse_rate = 0.3 * (1 + age_normalized * 0.5 + duration_normalized * 0.3)
adverse_events = np.random.poisson(adverse_rate * 3)  # Scale up

data['adverse_events'] = adverse_events

# ============================================================================
# STEP 4: **CAUSAL DROPOUT GENERATION** (THE KEY PART)
# ============================================================================

print("🎯 STEP 4: Generating CAUSAL dropout labels...")
print()

# Calculate RATES (not counts)
visit_rate = visits_completed / (days_in_trial / 30 + 1)  # Actual vs expected
adverse_event_rate = adverse_events / (days_in_trial + 1)  # Events per day

# Trial phase risk mapping (domain knowledge)
trial_phase_risk = {
    'Phase I': 0.2,    # Early, shorter commitment
    'Phase II': 0.5,   # Mid, longer commitment
    'Phase III': 0.8   # Late, longest commitment
}
phase_risk = np.array([trial_phase_risk[p] for p in data['trial_phase']])

# Treatment group risk
treatment_risk = {
    'Active': 0.1,     # Seeing benefits
    'Placebo': 0.4,    # No benefits, frustration
    'Control': 0.3     # Mixed
}
group_risk = np.array([treatment_risk[g] for g in data['treatment_group']])

# **CAUSAL RISK SCORE** (this creates learnable signal!)
risk_score = (
    0.35 * adverse_event_rate * 100 +        # High adverse events → dropout
    0.30 * (1 - visit_rate) +                 # Low compliance → dropout
    0.20 * phase_risk +                       # Advanced phase → dropout
    0.10 * group_risk +                       # Placebo/Control → dropout
    0.05 * (age_normalized < 0.3).astype(float)  # Young patients → dropout
)

# Add noise (realistic)
noise = np.random.normal(0, 0.15, n_rows)
risk_score = risk_score + noise

# Convert to binary dropout (probabilistic)
dropout_prob = 1 / (1 + np.exp(-3 * (risk_score - 0.5)))  # Sigmoid
dropout = (np.random.random(n_rows) < dropout_prob).astype(int)

data['dropout'] = dropout

# ============================================================================
# STEP 5: Generate dropout timing (for early/late classification)
# ============================================================================

dropout_day = np.zeros(n_rows)
early_dropout = np.zeros(n_rows)
late_dropout = np.zeros(n_rows)
dropout_30_days = np.zeros(n_rows)

for i in range(n_rows):
    if dropout[i] == 1:
        # Higher risk score → earlier dropout
        early_prob = min(0.8, risk_score[i])
        
        if np.random.random() < early_prob:
            # Early dropout (< 90 days)
            dropout_day[i] = np.random.randint(1, min(90, days_in_trial[i] + 1))
            early_dropout[i] = 1
            if dropout_day[i] <= 30:
                dropout_30_days[i] = 1
        else:
            # Late dropout (>= 90 days)
            if days_in_trial[i] >= 90:
                dropout_day[i] = np.random.randint(90, days_in_trial[i] + 1)
                late_dropout[i] = 1
            else:
                dropout_day[i] = np.random.randint(1, days_in_trial[i] + 1)
                early_dropout[i] = 1

data['dropout_day'] = dropout_day.astype(int)
data['early_dropout'] = early_dropout.astype(int)
data['late_dropout'] = late_dropout.astype(int)
data['dropout_30_days'] = dropout_30_days.astype(int)

# ============================================================================
# STEP 6: Create DataFrame and save
# ============================================================================

df = pd.DataFrame(data)

# Reorder columns
column_order = [
    'patient_id', 'age', 'gender', 'treatment_group', 'trial_phase',
    'days_in_trial', 'visits_completed', 'last_visit_day', 'adverse_events',
    'dropout', 'dropout_day', 'early_dropout', 'late_dropout', 'dropout_30_days'
]
df = df[column_order]

# Save
output_path = Path('data/raw/clinical_trials.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

# ============================================================================
# STEP 7: Verify learnable signal
# ============================================================================

print("✅ CAUSAL synthetic data generated")
print()
print("📊 Dropout Statistics:")
print(f"   Total Dropout Rate: {df['dropout'].mean():.1%}")
print(f"   Early Dropout Rate: {df['early_dropout'].mean():.1%}")
print(f"   Late Dropout Rate: {df['late_dropout'].mean():.1%}")
print(f"   30-Day Dropout Rate: {df['dropout_30_days'].mean():.1%}")
print()

# Check correlations (CRITICAL CHECK)
print("🔍 Checking for learnable signal...")
print("-" * 80)

# Calculate rate features
df['visit_rate'] = df['visits_completed'] / (df['days_in_trial'] / 30 + 1)
df['adverse_event_rate'] = df['adverse_events'] / (df['days_in_trial'] + 1)

# Check correlations
correlations = df[['visit_rate', 'adverse_event_rate', 'age', 'dropout']].corr()['dropout'].sort_values(ascending=False)
print(correlations)
print()

max_corr = correlations.abs().drop('dropout').max()
print(f"📈 Maximum absolute correlation: {max_corr:.4f}")

if max_corr > 0.15:
    print("✅ SUCCESS: Learnable signal exists (correlation > 0.15)")
    print("   → Models should achieve ROC-AUC > 0.65")
else:
    print("⚠️ WARNING: Weak signal (correlation < 0.15)")

print()
print("=" * 80)
print("💾 Saved to:", output_path)
print("=" * 80)
