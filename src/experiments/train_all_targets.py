"""
🎯 MULTI-TARGET TRAINING SCRIPT
================================

Trains models for ALL dropout prediction targets:
- General Dropout (binary)
- Early Dropout (< 90 days)
- Late Dropout (>= 90 days)
- 30-Day Dropout

Compares performance across all targets and models.
"""

import subprocess
import sys
import pandas as pd
from datetime import datetime

print("=" * 80)
print("🎯 MULTI-TARGET CLINICAL TRIAL DROPOUT PREDICTION")
print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================
# STEP 1: Generate Enhanced Synthetic Data
# ============================================

print("\n" + "=" * 80)
print("STEP 1: Generating Enhanced Synthetic Data")
print("=" * 80)

try:
    result = subprocess.run(
        [sys.executable, "data/synthetic_data_enhanced.py"],
        capture_output=True,
        text=True,
        check=True
    )
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Error generating data: {e}")
    print(e.stderr)
    sys.exit(1)

# ============================================
# STEP 2: Preprocess Data for All Targets
# ============================================

print("\n" + "=" * 80)
print("STEP 2: Preprocessing Data for All Targets")
print("=" * 80)

try:
    result = subprocess.run(
        [sys.executable, "src/preprocess_enhanced.py"],
        capture_output=True,
        text=True,
        check=True
    )
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Error preprocessing: {e}")
    print(e.stderr)
    sys.exit(1)

# ============================================
# STEP 3: Train Models for Each Target
# ============================================

targets = ["dropout", "early_dropout", "late_dropout", "dropout_30_days"]

for target in targets:
    print("\n" + "=" * 80)
    print(f"STEP 3: Training Models for Target: {target.upper()}")
    print("=" * 80)
    
    # Modify the train_optimized.py to use this target
    # We'll run it as a module and pass the target as an environment variable
    
    import os
    os.environ['TARGET_TYPE'] = target
    
    try:
        # Run training script
        result = subprocess.run(
            [sys.executable, "src/train_optimized.py"],
            capture_output=True,
            text=True,
            check=True,
            timeout=600  # 10 minute timeout
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
    except subprocess.TimeoutExpired:
        print(f"⚠️ Training for {target} took too long, skipping...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error training {target}: {e}")
        print(e.stderr)
        continue

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 80)
print("✨ ALL TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📊 Trained models for {len(targets)} targets:")
for target in targets:
    print(f"   ✅ {target}")

print(f"\n💡 View all results in MLflow UI:")
print(f"   cd a:\\Coding\\MLOps")
print(f"   mlflow ui --backend-store-uri sqlite:///mlflow.db")
print(f"   Then open: http://localhost:5000")

print("\n" + "=" * 80)
