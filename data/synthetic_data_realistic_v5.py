"""
data/synthetic_data_realistic_v5.py
====================================
REALISTIC SYNTHETIC DATA GENERATOR - V5

Based on human behavior patterns with:
- Patient archetypes (committed, fragile, busy, anxious, resilient)
- Probabilistic dropout
- Noisy, overlapping classes
- Temporal surprises
- Real-world complexity

This creates data that:
- Has NO clean class separation
- Some high-risk patients stay
- Some low-risk patients drop
- Includes late surprises
- ROC-AUC won't exceed ~0.75 (healthy)
"""

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

def clip01(x):
    """Clip value to [0, 1] range"""
    return max(0.0, min(1.0, x))


def sample_trial_phase():
    """Sample trial phase with realistic distribution"""
    return np.random.choice(
        ["Phase I", "Phase II", "Phase III"],
        p=[0.25, 0.35, 0.40]
    )


def sample_treatment_group():
    """Sample treatment group"""
    return np.random.choice(
        ["Active", "Control", "Placebo"],
        p=[0.40, 0.30, 0.30]
    )


def sample_gender():
    """Sample gender"""
    return np.random.choice(
        ["Male", "Female"],
        p=[0.50, 0.50]
    )


def generate_row(cfg):
    """
    Generate one realistic patient record
    
    Returns dict with all patient data including dropout outcome
    """
    # -------------------------------
    # Latent patient archetype
    # -------------------------------
    archetype = np.random.choice(
        ["committed", "fragile", "busy", "anxious", "resilient"],
        p=[0.35, 0.20, 0.20, 0.15, 0.10]
    )

    # -------------------------------
    # Basic demographics
    # -------------------------------
    age = np.random.randint(25, 80)
    days_in_trial = np.random.randint(30, 180)
    gender = sample_gender()
    treatment_group = sample_treatment_group()

    # -------------------------------
    # Visit behavior (archetype-driven)
    # -------------------------------
    base_visit_rate = {
        "committed": 0.9,
        "fragile": 0.75,
        "busy": 0.6,
        "anxious": 0.65,
        "resilient": 0.8
    }[archetype]

    visits_completed = max(
        1,
        int(np.random.normal(base_visit_rate * days_in_trial / 30, 1.2))
    )
    visits_completed = max(0, visits_completed)  # Can be 0 for very disengaged

    last_visit_day = np.random.randint(
        int(days_in_trial * 0.5),
        days_in_trial
    ) if visits_completed > 0 else 0

    # -------------------------------
    # Adverse events (non-linear)
    # -------------------------------
    adverse_events = np.random.poisson(
        lam={
            "committed": 0.6,
            "fragile": 2.8,
            "busy": 1.2,
            "anxious": 2.0,
            "resilient": 1.4
        }[archetype]
    )

    # -------------------------------
    # Derived behavioral metrics
    # -------------------------------
    visit_rate = visits_completed / max(1, days_in_trial / 30)
    adverse_event_rate = adverse_events / days_in_trial
    disengagement = max(0, 1 - visit_rate)
    days_since_last_visit = days_in_trial - last_visit_day

    # -------------------------------
    # Trial phase risk
    # -------------------------------
    trial_phase = sample_trial_phase()
    trial_phase_risk = cfg["trial_phase_risk"][trial_phase]

    # -------------------------------
    # Age-related risk (soft)
    # -------------------------------
    age_risk = clip01((age - 45) / 35)

    # -------------------------------
    # Late-stage surprise events
    # -------------------------------
    late_spike = np.random.binomial(1, 0.15)
    if late_spike:
        adverse_event_rate *= np.random.uniform(1.3, 2.0)
        disengagement *= np.random.uniform(1.1, 1.5)

    # -------------------------------
    # Continuous risk score (core)
    # -------------------------------
    risk_score = (
        cfg["risk_weights"]["adverse_event_rate"] * adverse_event_rate +
        cfg["risk_weights"]["disengagement"] * disengagement +
        cfg["risk_weights"]["trial_phase"] * trial_phase_risk +
        cfg["risk_weights"]["age"] * age_risk +
        0.10 * late_spike +
        np.random.normal(0, 0.08)
    )

    # -------------------------------
    # Recovery behavior (real-life)
    # -------------------------------
    if archetype == "resilient":
        risk_score *= np.random.uniform(0.6, 0.85)

    risk_score = clip01(risk_score)

    # -------------------------------
    # Probabilistic dropout
    # -------------------------------
    dropout = np.random.binomial(1, risk_score)

    # -------------------------------
    # Final row
    # -------------------------------
    return {
        "patient_id": "",  # Will be assigned later
        "age": age,
        "gender": gender,
        "treatment_group": treatment_group,
        "trial_phase": trial_phase,
        "days_in_trial": days_in_trial,
        "visits_completed": visits_completed,
        "last_visit_day": last_visit_day,
        "adverse_events": adverse_events,
        "dropout": dropout,
        "archetype": archetype,  # For analysis only
        "risk_score": risk_score  # For analysis only
    }


def main():
    """Generate realistic synthetic dataset"""
    
    # Configuration
    cfg = {
        "trial_phase_risk": {
            "Phase I": 0.2,
            "Phase II": 0.5,
            "Phase III": 0.8
        },
        "risk_weights": {
            "adverse_event_rate": 0.35,
            "disengagement": 0.30,
            "trial_phase": 0.20,
            "age": 0.15
        }
    }
    
    n_patients = 1000
    
    print("=" * 70)
    print("REALISTIC SYNTHETIC DATA GENERATOR - V5")
    print("=" * 70)
    print(f"Generating {n_patients} patients with human behavior patterns...")
    print()
    
    # Generate data
    data = []
    for i in range(n_patients):
        row = generate_row(cfg)
        row['patient_id'] = f'P-{str(i+1).zfill(4)}'
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns (analysis columns at end)
    df = df[['patient_id', 'age', 'gender', 'treatment_group', 'trial_phase',
             'days_in_trial', 'visits_completed', 'last_visit_day',
             'adverse_events', 'dropout', 'archetype', 'risk_score']]
    
    # Save to CSV
    output_path = 'data/raw/clinical_trials_realistic_v5.csv'
    df.to_csv(output_path, index=False)
    
    # Print statistics
    print("✅ Data generation complete!")
    print(f"📁 Saved to: {output_path}")
    print()
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total patients: {len(df)}")
    print(f"Dropout rate: {df['dropout'].mean():.2%}")
    print()
    
    print("Patient Archetypes:")
    print(df['archetype'].value_counts().sort_index())
    print()
    
    print("Dropout by Archetype:")
    archetype_stats = df.groupby('archetype')['dropout'].agg(['count', 'sum', 'mean'])
    archetype_stats.columns = ['Total', 'Dropouts', 'Dropout Rate']
    print(archetype_stats)
    print()
    
    print("Trial Phase Distribution:")
    print(df['trial_phase'].value_counts().sort_index())
    print()
    
    print("Dropout by Trial Phase:")
    phase_stats = df.groupby('trial_phase')['dropout'].agg(['count', 'sum', 'mean'])
    phase_stats.columns = ['Total', 'Dropouts', 'Dropout Rate']
    print(phase_stats)
    print()
    
    print("Treatment Group Distribution:")
    print(df['treatment_group'].value_counts().sort_index())
    print()
    
    print("=" * 70)
    print("RISK SCORE ANALYSIS")
    print("=" * 70)
    print(f"Mean risk score: {df['risk_score'].mean():.3f}")
    print(f"Std risk score: {df['risk_score'].std():.3f}")
    print(f"Min risk score: {df['risk_score'].min():.3f}")
    print(f"Max risk score: {df['risk_score'].max():.3f}")
    print()
    
    # Risk score quartiles
    df['risk_quartile'] = pd.qcut(df['risk_score'], q=4, labels=['Q1-Low', 'Q2-Medium', 'Q3-High', 'Q4-Very High'])
    print("Dropout by Risk Quartile:")
    quartile_stats = df.groupby('risk_quartile')['dropout'].agg(['count', 'sum', 'mean'])
    quartile_stats.columns = ['Total', 'Dropouts', 'Dropout Rate']
    print(quartile_stats)
    print()
    
    print("=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)
    print(df[['age', 'days_in_trial', 'visits_completed', 'adverse_events']].describe())
    print()
    
    print("=" * 70)
    print("✅ REALISTIC DATA GENERATED")
    print("=" * 70)
    print("Characteristics:")
    print("  ✓ Human behavior archetypes")
    print("  ✓ Probabilistic dropout")
    print("  ✓ Overlapping classes")
    print("  ✓ Temporal surprises (15% late spikes)")
    print("  ✓ Noisy, real-world complexity")
    print()
    print(f"Expected model performance: ROC-AUC ~0.65-0.75 (healthy)")


if __name__ == "__main__":
    main()
