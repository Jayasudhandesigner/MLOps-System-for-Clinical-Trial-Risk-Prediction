"""
monitoring/build_reference.py
==============================
BUILD REFERENCE DATASET FOR DRIFT DETECTION

Purpose: Create a "snapshot" of what normal data looks like.
This reference is compared against live data to detect drift.

Usage:
    python monitoring/build_reference.py

Output:
    monitoring/reference.csv

Version with DVC:
    dvc add monitoring/reference.csv
    git commit -m "monitoring: add reference dataset"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def build_reference(
    input_path: str = "data/raw/clinical_trials_realistic_v5.csv",
    output_path: str = "monitoring/reference.csv",
    sample_size: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Build reference dataset from training data.
    
    Args:
        input_path: Path to training data
        output_path: Path to save reference dataset
        sample_size: Number of samples (use all if less)
        random_state: Random seed for reproducibility
    
    Returns:
        Reference DataFrame
    """
    print("=" * 70)
    print("🔧 BUILDING REFERENCE DATASET FOR DRIFT DETECTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Load training data
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Training data not found: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"📁 Loaded {len(df)} records from {input_path}")
    
    # Sample if dataset is larger than sample_size
    if len(df) > sample_size:
        reference = df.sample(n=sample_size, random_state=random_state)
        print(f"📊 Sampled {sample_size} records (random_state={random_state})")
    else:
        reference = df.copy()
        print(f"📊 Using all {len(df)} records (smaller than sample_size)")
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save reference
    reference.to_csv(output_path, index=False)
    print(f"✅ Saved reference dataset to {output_path}")
    
    # Statistics
    print()
    print("-" * 70)
    print("REFERENCE DATASET STATISTICS")
    print("-" * 70)
    print(f"  Total records:    {len(reference)}")
    print(f"  Dropout rate:     {reference['dropout'].mean():.2%}")
    print(f"  Mean age:         {reference['age'].mean():.1f}")
    print(f"  Mean days:        {reference['days_in_trial'].mean():.1f}")
    print()
    
    # Feature distributions (for reference)
    print("Feature Distribution Summary:")
    numeric_cols = ['age', 'days_in_trial', 'visits_completed', 'adverse_events']
    for col in numeric_cols:
        print(f"  {col:20s}: mean={reference[col].mean():.2f}, std={reference[col].std():.2f}")
    
    print()
    print("=" * 70)
    print("✅ REFERENCE DATASET READY FOR DRIFT DETECTION")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. dvc add monitoring/reference.csv")
    print("  2. git commit -m 'monitoring: add reference dataset'")
    print()
    
    return reference


if __name__ == "__main__":
    build_reference()
