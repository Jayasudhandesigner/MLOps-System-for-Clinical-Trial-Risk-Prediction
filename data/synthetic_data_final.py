"""
data/synthetic_data_final.py
=============================
FINAL BALANCED SYNTHETIC DATA - Production Quality

Creates data with CLEAR SIGNAL that models can actually learn from.
Balance between realism and learnability.
"""

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

def generate_final_dataset(n=1000):
    """
    Generate production-quality synthetic data with clear signal.
    
    Strategy:
    - Start with dropout probability based on clear risk factors
    - Add moderate noise (not too much!)
    - Ensure learnable patterns exist
    """
    data = []
    
    for i in range(n):
        # Base demographics
        age = np.random.randint(25, 85)
        gender = np.random.choice(['Male', 'Female'])
        trial_phase = np.random.choice(['Phase I', 'Phase II', 'Phase III'], p=[0.25, 0.35, 0.40])
        treatment_group = np.random.choice(['Active', 'Control', 'Placebo'], p=[0.40, 0.30, 0.30])
        
        # Duration in trial
        days_in_trial = np.random.randint(30, 200)
        
        # ==========================================
        # RISK CALCULATION (Clear Signal)
        # ==========================================
        
        # Age risk (older = higher risk)
        age_risk = (age - 25) / 60  #0 to 1 normalized
        
        # Phase risk (later phase = higher dropout)
        phase_risk_map = {'Phase I': 0.2, 'Phase II': 0.5, 'Phase III': 0.8}
        phase_risk = phase_risk_map[trial_phase]
        
        # Treatment risk (placebo = highest)
        treatment_risk_map = {'Active': 0.1, 'Control': 0.3, 'Placebo': 0.5}
        treatment_risk = treatment_risk_map[treatment_group]
        
        # Calculate base dropout probability
        base_dropout_prob = (
            0.30 * age_risk +
            0.35 * phase_risk +
            0.25 * treatment_risk +
            0.10 * np.random.random()  # Small random component
        )
        
        # Clip to valid range
        base_dropout_prob = max(0.05, min(0.90, base_dropout_prob))
        
        # ==========================================
        # GENERATE FEATURES BASED ON RISK
        # ==========================================
        
        # Visits (lower compliance for higher risk)
        expected_visits = days_in_trial // 30
        compliance_factor = 1 - base_dropout_prob  # Lower risk = better compliance
        visits_completed = max(0, int(expected_visits * compliance_factor * np.random.uniform(0.7, 1.3)))
        
        # Last visit day (higher risk = longer since last visit)
        if visits_completed > 0:
            recency_factor = base_dropout_prob
            time_since_visit = int(days_in_trial * recency_factor * np.random.uniform(0.3, 0.8))
            last_visit_day = max(0, days_in_trial - time_since_visit)
        else:
            last_visit_day = 0
        
        # Adverse events (Poisson distribution, higher for high risk)
        adverse_lambda = base_dropout_prob * 5 + 0.5  # 0.5 to 5
        adverse_events = np.random.poisson(adverse_lambda)
        
        # ==========================================
        # FINAL DROPOUT DECISION
        # ==========================================
        
        # Add slight randomness but keep signal strong
        final_dropout_prob = base_dropout_prob + np.random.normal(0, 0.10)
        final_dropout_prob = max(0.0, min(1.0, final_dropout_prob))
        
        dropout = 1 if np.random.random() < final_dropout_prob else 0
        
        # ==========================================
        # DERIVED COLUMNS
        # ==========================================
        
        if dropout == 1:
            # Dropout day somewhere in trial
            dropout_day = np.random.randint(max(1, days_in_trial // 3), days_in_trial)
            early_dropout = 1 if dropout_day <= 90 else 0
            late_dropout = 1 if dropout_day > 90 else 0
            dropout_30_days = 1 if dropout_day <= 30 else 0
        else:
            dropout_day = days_in_trial
            early_dropout = 0
            late_dropout = 0
            dropout_30_days = 0
        
        # ==========================================
        # BUILD ROW
        # ==========================================
        
        data.append({
            'patient_id': f'P-{str(i+1).zfill(4)}',
            'age': age,
            'gender': gender,
            'treatment_group': treatment_group,
            'trial_phase': trial_phase,
            'days_in_trial': days_in_trial,
            'visits_completed': visits_completed,
            'last_visit_day': last_visit_day,
            'adverse_events': adverse_events,
            'dropout': dropout,
            'dropout_day': dropout_day,
            'early_dropout': early_dropout,
            'late_dropout': late_dropout,
            'dropout_30_days': dropout_30_days
        })
    
    return pd.DataFrame(data)


def main():
    print("="*70)
    print("GENERATING FINAL PRODUCTION-QUALITY DATASET")
    print("="*70)
    print()
    
    # Generate data
    df = generate_final_dataset(n=1000)
    
    # Save
    output_path = 'data/raw/clinical_trials_realistic_v5.csv'
    df.to_csv(output_path, index=False)
    
    # Stats
    print(f"✅ Generated {len(df)} patients")
    print(f"📁 Saved to: {output_path}")
    print()
    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Dropout rate: {df['dropout'].mean():.2%}")
    print()
    
    print("Dropout by Trial Phase:")
    phase_stats = df.groupby('trial_phase')['dropout'].agg(['count', 'sum', lambda x: x.mean()])
    phase_stats.columns = ['Total', 'Dropouts', 'Rate']
    print(phase_stats)
    print()
    
    print("Dropout by Treatment:")
    treatment_stats = df.groupby('treatment_group')['dropout'].agg(['count', 'sum', lambda x: x.mean()])
    treatment_stats.columns = ['Total', 'Dropouts', 'Rate']
    print(treatment_stats)
    print()
    
    print("Dropout by Age Group:")
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 100], labels=['Young', 'Middle', 'Senior'])
    age_stats = df.groupby('age_group')['dropout'].agg(['count', 'sum', lambda x: x.mean()])
    age_stats.columns = ['Total', 'Dropouts', 'Rate']
    print(age_stats)
    print()
    
    print("="*70)
    print("FEATURE STATISTICS")
    print("="*70)
    print(df[['age', 'days_in_trial', 'visits_completed', 'adverse_events']].describe())
    print()
    
    print("="*70)
    print("✅ DATA QUALITY CHECK")
    print("="*70)
    print("Signal strength indicators:")
    print(f"  - Dropout rate variance across phases: {df.groupby('trial_phase')['dropout'].mean().std():.3f}")
    print(f"  - Dropout rate variance across treatment: {df.groupby('treatment_group')['dropout'].mean().std():.3f}")
    print(f"  - Expected ROC-AUC: 0.65-0.75 (good learnable signal)")
    print()


if __name__ == "__main__":
    main()
