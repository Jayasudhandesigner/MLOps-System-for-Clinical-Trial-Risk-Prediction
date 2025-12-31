"""
tests/test_features.py
======================
Feature Engineering Validation Tests

Purpose: Ensure feature transformations are correct.
Prevents: NaN propagation, invalid feature ranges, broken pipelines.

Run: pytest tests/test_features.py -v
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path for feature module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


DATA_PATH = Path("data/raw/clinical_trials_realistic_v5.csv")


class TestFeatureEngineering:
    """Feature engineering validation tests."""
    
    @pytest.fixture(autouse=True)
    def load_data(self):
        """Load data once for all tests."""
        if not DATA_PATH.exists():
            pytest.skip(f"Data file not found: {DATA_PATH}")
        self.df = pd.read_csv(DATA_PATH)
    
    def test_visits_completed_non_negative(self):
        """Visits completed must be non-negative integers."""
        assert (self.df["visits_completed"] >= 0).all(), \
            "Found negative visits_completed values"
    
    def test_adverse_events_non_negative(self):
        """Adverse events count must be non-negative."""
        assert (self.df["adverse_events"] >= 0).all(), \
            "Found negative adverse_events values"
    
    def test_last_visit_day_consistency(self):
        """Last visit day must not exceed days in trial."""
        valid = self.df["last_visit_day"] <= self.df["days_in_trial"]
        assert valid.all(), \
            f"Found {(~valid).sum()} records where last_visit_day > days_in_trial"
    
    def test_trial_phase_categories(self):
        """Trial phase must be one of expected categories."""
        valid_phases = {"Phase I", "Phase II", "Phase III"}
        actual_phases = set(self.df["trial_phase"].unique())
        assert actual_phases.issubset(valid_phases), \
            f"Invalid trial phases: {actual_phases - valid_phases}"
    
    def test_treatment_group_categories(self):
        """Treatment group must be one of expected categories."""
        valid_groups = {"Active", "Control", "Placebo"}
        actual_groups = set(self.df["treatment_group"].unique())
        assert actual_groups.issubset(valid_groups), \
            f"Invalid treatment groups: {actual_groups - valid_groups}"
    
    def test_gender_categories(self):
        """Gender must be one of expected categories."""
        valid_genders = {"Male", "Female"}
        actual_genders = set(self.df["gender"].unique())
        assert actual_genders.issubset(valid_genders), \
            f"Invalid gender values: {actual_genders - valid_genders}"
    
    def test_no_nan_in_numeric_features(self):
        """Numeric features must not contain NaN values."""
        numeric_cols = ["age", "days_in_trial", "visits_completed", 
                        "adverse_events", "dropout"]
        for col in numeric_cols:
            if col in self.df.columns:
                nan_count = self.df[col].isna().sum()
                assert nan_count == 0, \
                    f"Column '{col}' contains {nan_count} NaN values"
    
    def test_dropout_day_logic(self):
        """Dropout day must be consistent with dropout status."""
        # For dropouts, dropout_day should be less than or equal to days_in_trial
        dropouts = self.df[self.df["dropout"] == 1]
        if len(dropouts) > 0 and "dropout_day" in self.df.columns:
            valid = dropouts["dropout_day"] <= dropouts["days_in_trial"]
            assert valid.all(), \
                f"Found {(~valid).sum()} dropouts with invalid dropout_day"


class TestFeatureCorrelations:
    """Test that features have expected correlations with target."""
    
    @pytest.fixture(autouse=True)
    def load_data(self):
        """Load data once for all tests."""
        if not DATA_PATH.exists():
            pytest.skip(f"Data file not found: {DATA_PATH}")
        self.df = pd.read_csv(DATA_PATH)
    
    def test_adverse_events_correlation(self):
        """Adverse events should correlate positively with dropout."""
        # Higher adverse events -> more likely to dropout
        high_ae = self.df[self.df["adverse_events"] >= 3]["dropout"].mean()
        low_ae = self.df[self.df["adverse_events"] <= 1]["dropout"].mean()
        
        assert high_ae > low_ae, \
            f"Expected higher dropout with more adverse events: " \
            f"high_ae={high_ae:.2%}, low_ae={low_ae:.2%}"
