"""
src/core/features.py
=====================
PRODUCTION FEATURE ENGINEERING MODULE (CAUSAL SIGNAL VERSION)

Key principle: RATES create separation, COUNTS don't.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def engineer_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create RATE features (critical for learnable signal).
    
    Why rates? They normalize for time and create separation.
    
    Features created:
    - visit_rate: Actual vs expected visit completion
    - adverse_event_rate: Events per day
    - time_since_last_visit: Engagement gap
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame with rate features
    """
    df = df.copy()
    
    # Visit rate (compliance measure)
    # Expected: ~1 visit per 30 days
    expected_visits = df['days_in_trial'] / 30 + 1
    df['visit_rate'] = df['visits_completed'] / expected_visits
    
    # Adverse event rate (normalized by time)
    df['adverse_event_rate'] = df['adverse_events'] / (df['days_in_trial'] + 1)
    
    # Time since last visit (engagement gap)
    df['time_since_last_visit'] = df['days_in_trial'] - df['last_visit_day']
    
    logger.info("✅ Created 3 rate features")
    return df


def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create INTERACTION features (capture compound effects).
    
    Key insight: Dropout isn't caused by one thing, it's caused by PRESSURE.
    
    Features created:
    - burden: High adverse events × Low visit compliance
    - age_adverse_risk: Older patients with adverse events
    
    Args:
        df: DataFrame with rate features
        
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    # Burden: Compound pressure from adverse events + poor compliance
    # High adverse events AND low visits → very high dropout risk
    df['burden'] = df['adverse_event_rate'] * (1 - df['visit_rate'])
    
    # Age-adverse risk: Older patients tolerate adverse events less
    df['age_adverse_risk'] = (df['age'] / 85) * df['adverse_event_rate']
    
    logger.info("✅ Created 2 interaction features")
    return df


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode DOMAIN KNOWLEDGE (ordinal relationships).
    
    Not all categories are equal - encode known risk patterns.
    
    Features created:
    - trial_phase_risk: Phase I < Phase II < Phase III
    - treatment_risk: Active < Control < Placebo
    
    Args:
        df: DataFrame with categorical features
        
    Returns:
        DataFrame with domain-encoded features
    """
    df = df.copy()
    
    # Trial phase risk (longer trials = higher dropout)
    trial_phase_risk_map = {
        'Phase I': 0.2,
        'Phase II': 0.5,
        'Phase III': 0.8
    }
    df['trial_phase_risk'] = df['trial_phase'].map(trial_phase_risk_map)
    
    # Treatment group risk (placebo/control higher dropout)
    treatment_risk_map = {
        'Active': 0.1,      # Seeing benefits
        'Control': 0.3,     # Mixed
        'Placebo': 0.4      # No benefits
    }
    df['treatment_risk'] = df['treatment_group'].map(treatment_risk_map)
    
    logger.info("✅ Created 2 domain knowledge features")
    return df


if __name__ == "__main__":
    # Test feature engineering
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    test_df = pd.DataFrame({
        'days_in_trial': [90, 180, 270],
        'visits_completed': [2, 5, 7],
        'adverse_events': [1, 3, 5],
        'last_visit_day': [75, 150, 240],
        'age': [45, 65, 75],
        'trial_phase': ['Phase I', 'Phase II', 'Phase III'],
        'treatment_group': ['Active', 'Control', 'Placebo']
    })
    
    result = engineer_rate_features(test_df)
    result = engineer_interaction_features(result)
    result = engineer_domain_features(result)
    
    print("\n✅ Feature engineering test passed")
    print("\nGenerated features:")
    print(result[['visit_rate', 'adverse_event_rate', 'burden', 'trial_phase_risk']].head())
