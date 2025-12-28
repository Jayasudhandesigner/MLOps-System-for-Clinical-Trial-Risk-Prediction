"""
data/synthetic_data_balanced_v4.py
===================================
BALANCED SYNTHETIC DATA GENERATOR - V4

Creates 5 risk categories with equal distribution:
- Very Low Risk (20%)
- Low Risk (20%)
- Medium Risk (20%)
- High Risk (20%)
- Very High Risk (20%)

This ensures the model learns to discriminate across all risk levels.
"""

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Total patients
N_TOTAL = 1000
N_PER_RISK = 200  # Equal distribution

def generate_very_low_risk(n):
    """
    VERY LOW RISK: Young, active treatment, Phase I, few adverse events
    Dropout probability: 5-15%
    """
    data = []
    for _ in range(n):
        age = np.random.randint(25, 45)
        treatment_group = np.random.choice(['Active'], p=[1.0])
        trial_phase = np.random.choice(['Phase I'], p=[1.0])
        days_in_trial = np.random.randint(30, 90)
        
        # High compliance
        expected_visits = days_in_trial // 30
        visits_completed = max(expected_visits, np.random.randint(expected_visits, expected_visits + 2))
        last_visit_day = days_in_trial - np.random.randint(0, 15)
        
        # Very few adverse events
        adverse_events = np.random.randint(0, 2)
        
        # Low dropout probability (5-15%)
        dropout = 1 if np.random.random() < 0.10 else 0
        
        data.append({
            'age': age,
            'gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
            'treatment_group': treatment_group,
            'trial_phase': trial_phase,
            'days_in_trial': days_in_trial,
            'visits_completed': visits_completed,
            'last_visit_day': last_visit_day,
            'adverse_events': adverse_events,
            'dropout': dropout,
            'risk_category': 'Very Low'
        })
    
    return data


def generate_low_risk(n):
    """
    LOW RISK: Middle-aged, active/control, Phase I/II, moderate compliance
    Dropout probability: 15-25%
    """
    data = []
    for _ in range(n):
        age = np.random.randint(40, 55)
        treatment_group = np.random.choice(['Active', 'Control'], p=[0.7, 0.3])
        trial_phase = np.random.choice(['Phase I', 'Phase II'], p=[0.6, 0.4])
        days_in_trial = np.random.randint(60, 120)
        
        # Good compliance
        expected_visits = days_in_trial // 30
        visits_completed = max(1, np.random.randint(expected_visits - 1, expected_visits + 1))
        last_visit_day = days_in_trial - np.random.randint(5, 25)
        
        # Few adverse events
        adverse_events = np.random.randint(0, 4)
        
        # Low-medium dropout probability (15-25%)
        dropout = 1 if np.random.random() < 0.20 else 0
        
        data.append({
            'age': age,
            'gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
            'treatment_group': treatment_group,
            'trial_phase': trial_phase,
            'days_in_trial': days_in_trial,
            'visits_completed': visits_completed,
            'last_visit_day': last_visit_day,
            'adverse_events': adverse_events,
            'dropout': dropout,
            'risk_category': 'Low'
        })
    
    return data


def generate_medium_risk(n):
    """
    MEDIUM RISK: Older adults, control/placebo, Phase II, mixed compliance
    Dropout probability: 30-40%
    """
    data = []
    for _ in range(n):
        age = np.random.randint(50, 65)
        treatment_group = np.random.choice(['Control', 'Placebo', 'Active'], p=[0.4, 0.4, 0.2])
        trial_phase = np.random.choice(['Phase II', 'Phase III'], p=[0.6, 0.4])
        days_in_trial = np.random.randint(80, 150)
        
        # Moderate compliance
        expected_visits = days_in_trial // 30
        visits_completed = max(1, np.random.randint(expected_visits - 2, expected_visits))
        last_visit_day = days_in_trial - np.random.randint(15, 45)
        
        # Moderate adverse events
        adverse_events = np.random.randint(2, 6)
        
        # Medium dropout probability (30-40%)
        dropout = 1 if np.random.random() < 0.35 else 0
        
        data.append({
            'age': age,
            'gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
            'treatment_group': treatment_group,
            'trial_phase': trial_phase,
            'days_in_trial': days_in_trial,
            'visits_completed': visits_completed,
            'last_visit_day': last_visit_day,
            'adverse_events': adverse_events,
            'dropout': dropout,
            'risk_category': 'Medium'
        })
    
    return data


def generate_high_risk(n):
    """
    HIGH RISK: Seniors, placebo, Phase III, poor compliance
    Dropout probability: 50-60%
    """
    data = []
    for _ in range(n):
        age = np.random.randint(60, 75)
        treatment_group = np.random.choice(['Placebo', 'Control'], p=[0.7, 0.3])
        trial_phase = np.random.choice(['Phase III', 'Phase II'], p=[0.8, 0.2])
        days_in_trial = np.random.randint(100, 180)
        
        # Poor compliance
        expected_visits = days_in_trial // 30
        visits_completed = max(1, np.random.randint(1, max(2, expected_visits - 1)))
        last_visit_day = days_in_trial - np.random.randint(30, 70)
        
        # Many adverse events
        adverse_events = np.random.randint(4, 10)
        
        # High dropout probability (50-60%)
        dropout = 1 if np.random.random() < 0.55 else 0
        
        data.append({
            'age': age,
            'gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
            'treatment_group': treatment_group,
            'trial_phase': trial_phase,
            'days_in_trial': days_in_trial,
            'visits_completed': visits_completed,
            'last_visit_day': last_visit_day,
            'adverse_events': adverse_events,
            'dropout': dropout,
            'risk_category': 'High'
        })
    
    return data


def generate_very_high_risk(n):
    """
    VERY HIGH RISK: Elderly, placebo, Phase III, very poor compliance
    Dropout probability: 70-85%
    """
    data = []
    for _ in range(n):
        age = np.random.randint(70, 85)
        treatment_group = np.random.choice(['Placebo'], p=[1.0])
        trial_phase = np.random.choice(['Phase III'], p=[1.0])
        days_in_trial = np.random.randint(120, 200)
        
        # Very poor compliance
        expected_visits = days_in_trial // 30
        visits_completed = np.random.randint(0, max(1, expected_visits - 2))
        last_visit_day = np.random.randint(0, days_in_trial // 2) if visits_completed > 0 else 0
        
        # Severe adverse events
        adverse_events = np.random.randint(6, 16)
        
        # Very high dropout probability (70-85%)
        dropout = 1 if np.random.random() < 0.77 else 0
        
        data.append({
            'age': age,
            'gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
            'treatment_group': treatment_group,
            'trial_phase': trial_phase,
            'days_in_trial': days_in_trial,
            'visits_completed': visits_completed,
            'last_visit_day': last_visit_day,
            'adverse_events': adverse_events,
            'dropout': dropout,
            'risk_category': 'Very High'
        })
    
    return data


def main():
    """Generate balanced synthetic dataset"""
    print("Generating Balanced Synthetic Clinical Trial Data (V4)...")
    print(f"Total patients: {N_TOTAL}")
    print(f"Per risk category: {N_PER_RISK}")
    print()
    
    # Generate each risk category
    very_low = generate_very_low_risk(N_PER_RISK)
    low = generate_low_risk(N_PER_RISK)
    medium = generate_medium_risk(N_PER_RISK)
    high = generate_high_risk(N_PER_RISK)
    very_high = generate_very_high_risk(N_PER_RISK)
    
    # Combine all
    all_data = very_low + low + medium + high + very_high
    
    # Shuffle randomly to mix risk levels
    np.random.shuffle(all_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Add patient IDs
    df['patient_id'] = [f'P-{str(i).zfill(4)}' for i in range(1, len(df) + 1)]
    
    # Reorder columns
    df = df[['patient_id', 'age', 'gender', 'treatment_group', 'trial_phase',
             'days_in_trial', 'visits_completed', 'last_visit_day',
             'adverse_events', 'dropout', 'risk_category']]
    
    # Save to CSV
    output_path = 'data/raw/clinical_trials_balanced_v4.csv'
    df.to_csv(output_path, index=False)
    
    # Print statistics
    print("✅ Data generation complete!")
    print(f"📁 Saved to: {output_path}")
    print()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total patients: {len(df)}")
    print(f"Dropout rate: {df['dropout'].mean():.2%}")
    print()
    print("Risk Category Distribution:")
    print(df['risk_category'].value_counts().sort_index())
    print()
    print("Dropout by Risk Category:")
    print(df.groupby('risk_category')['dropout'].agg(['count', 'sum', 'mean']))
    print()
    print("=" * 60)
    print("FEATURE STATISTICS")
    print("=" * 60)
    print(df.describe())


if __name__ == "__main__":
    main()
