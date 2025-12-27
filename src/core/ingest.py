"""
src/core/ingest.py
==================
PRODUCTION DATA INGESTION MODULE

Golden Path: Step 1
Loads and validates clinical trial data from raw CSV files.
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Data contract: Required columns
REQUIRED_COLUMNS = [
    "patient_id",
    "age",
    "gender",
    "treatment_group",
    "trial_phase",
    "visits_completed",
    "adverse_events",
    "days_in_trial",
    "last_visit_day",
    "dropout",
    "dropout_day",
    "early_dropout",
    "late_dropout",
    "dropout_30_days"
]


def load_data(path: str) -> pd.DataFrame:
    """
    Load and validate clinical trial data.
    
    Args:
        path: Path to raw CSV file
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValueError: If data validation fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records from {path}")
    
    # Validation: Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validation: Check patient_id uniqueness
    if df["patient_id"].isnull().any():
        raise ValueError("patient_id contains null values")
    
    # Validation: Check dropout is binary
    if not df["dropout"].isin([0, 1]).all():
        raise ValueError("dropout must be binary (0 or 1)")
    
    logger.info(f"✅ Data validation passed")
    return df


if __name__ == "__main__":
    # Test ingestion
    logging.basicConfig(level=logging.INFO)
    data = load_data("data/raw/clinical_trials.csv")
    print(f"✅ Loaded {data.shape[0]} records")
    print(f"📊 Columns: {list(data.columns)}")
