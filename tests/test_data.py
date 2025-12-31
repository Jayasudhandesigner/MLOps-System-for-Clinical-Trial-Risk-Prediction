"""
tests/test_data.py
==================
Data Validation Tests for MLOps Pipeline

Purpose: Ensure data quality gates BEFORE training.
Prevents: Garbage data, empty datasets, corrupted files.

Run: pytest tests/test_data.py -v
"""

import pandas as pd
import pytest
from pathlib import Path


# Data file path - adjust based on your environment
DATA_PATH = Path("data/raw/clinical_trials_realistic_v5.csv")


class TestDataQuality:
    """Data quality validation tests."""
    
    @pytest.fixture(autouse=True)
    def load_data(self):
        """Load data once for all tests."""
        if not DATA_PATH.exists():
            pytest.skip(f"Data file not found: {DATA_PATH}")
        self.df = pd.read_csv(DATA_PATH)
    
    def test_no_empty_data(self):
        """Dataset must have sufficient samples for training."""
        assert len(self.df) > 100, \
            f"Dataset too small: {len(self.df)} rows (minimum 100 required)"
    
    def test_target_distribution(self):
        """Dropout rate must be within realistic bounds (5-50%)."""
        rate = self.df["dropout"].mean()
        assert 0.05 < rate < 0.50, \
            f"Dropout rate {rate:.2%} outside acceptable range (5-50%)"
    
    def test_no_missing_patient_ids(self):
        """All patients must have valid IDs."""
        assert self.df["patient_id"].notna().all(), \
            "Missing patient IDs detected"
    
    def test_target_is_binary(self):
        """Target variable must be binary (0 or 1)."""
        unique_values = set(self.df["dropout"].unique())
        assert unique_values.issubset({0, 1}), \
            f"Target contains non-binary values: {unique_values}"
    
    def test_required_columns_present(self):
        """All required columns must exist in dataset."""
        required_cols = [
            "patient_id", "age", "gender", "treatment_group",
            "trial_phase", "days_in_trial", "visits_completed",
            "adverse_events", "dropout"
        ]
        missing = [col for col in required_cols if col not in self.df.columns]
        assert not missing, f"Missing required columns: {missing}"
    
    def test_no_duplicate_patient_ids(self):
        """Patient IDs must be unique."""
        duplicates = self.df["patient_id"].duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate patient IDs"
    
    def test_age_in_valid_range(self):
        """Patient ages must be realistic (18-100)."""
        assert (self.df["age"] >= 18).all(), "Found patients younger than 18"
        assert (self.df["age"] <= 100).all(), "Found patients older than 100"
    
    def test_days_in_trial_positive(self):
        """Days in trial must be positive."""
        assert (self.df["days_in_trial"] > 0).all(), \
            "Found non-positive days_in_trial values"
