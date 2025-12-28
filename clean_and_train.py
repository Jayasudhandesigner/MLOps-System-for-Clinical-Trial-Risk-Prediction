"""
clean_and_train.py
==================
COMPREHENSIVE CLEANUP AND FRESH MODEL TRAINING

Executes in one sweep:
1. Clears ALL cache and generated artifacts
2. Runs complete pipeline from scratch
"""

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("\n" + "="*80)
print("COMPREHENSIVE CACHE CLEANUP & FRESH MODEL TRAINING")
print("="*80 + "\n")

# ============================================================================
# PHASE 1: CACHE CLEANUP
# ============================================================================

print("PHASE 1: CLEARING ALL CACHE\n")

# Clear Python cache
print("Clearing Python cache...")
for pycache in PROJECT_ROOT.rglob('__pycache__'):
    try:
        shutil.rmtree(pycache)
        print(f"  ✅ Removed: {pycache.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

# Clear .pyc files
for pyc in PROJECT_ROOT.rglob('*.pyc'):
    try:
        pyc.unlink()
    except:
        pass

# Clear processed data
processed_dir = PROJECT_ROOT / "data" / "processed"
if processed_dir.exists():
    print("\nClearing processed data...")
    for item in processed_dir.iterdir():
        if item.name != '.gitkeep':
            try:
                item.unlink()
                print(f"  ✅ Removed: {item.name}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")

# Clear models
models_dir = PROJECT_ROOT / "models"
if models_dir.exists():
    print("\nClearing models...")
    for item in models_dir.iterdir():
        if item.name != '.gitkeep':
            try:
                item.unlink()
                print(f"  ✅ Removed: {item.name}")
            except:
                pass

# Clear logs
logs_dir = PROJECT_ROOT / "logs"
if logs_dir.exists():
    print("\nClearing logs...")
    for item in logs_dir.iterdir():
        if item.name != '.gitkeep':
            try:
                item.unlink()
                print(f"  ✅ Removed: {item.name}")
            except:
                pass

# Clear MLflow
mlruns_dir = PROJECT_ROOT / "mlruns"
if mlruns_dir.exists():
    try:
        shutil.rmtree(mlruns_dir)
        print("\n  ✅ Removed: mlruns/")
    except:
        pass

mlflow_db = PROJECT_ROOT / "mlflow.db"
if mlflow_db.exists():
    try:
        mlflow_db.unlink()
        print("  ✅ Removed: mlflow.db")
    except:
        pass

print("\n" + "="*80)
print("CACHE CLEANUP COMPLETE!")
print("="*80 + "\n")

# ============================================================================
# PHASE 2: RUN FULL PIPELINE
# ============================================================================

print("PHASE 2: RUNNING FULL PIPELINE FROM SCRATCH\n")
print("This will:")
print("  1. Generate 1000 synthetic patients")
print("  2. Engineer causal features")
print("  3. Preprocess data")
print("  4. Train 3 models (Logistic, XGBoost, LightGBM)")
print("\nThis takes ~3-5 minutes...\n")
print("="*80 + "\n")

# Run the pipeline
os.chdir(PROJECT_ROOT)
exit_code = os.system("python pipelines/local_pipeline.py")

if exit_code != 0:
    print(f"\n❌ Pipeline failed with exit code: {exit_code}")
    sys.exit(1)

# ============================================================================
# PHASE 3: VALIDATION
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: VALIDATION")
print("="*80 + "\n")

# Check artifacts
required_files = [
    "data/raw/clinical_trials_realistic_v5.csv",
    "data/processed/clinical_trials_dropout.csv",
    "data/processed/preprocessor_dropout_v3_causal.pkl",
    "mlflow.db"
]

all_good = True
for file_path in required_files:
    full_path = PROJECT_ROOT / file_path
    if full_path.exists():
        size_mb = full_path.stat().st_size / 1024 / 1024
        print(f"✅ {file_path} ({size_mb:.2f} MB)")
    else:
        print(f"❌ MISSING: {file_path}")
        all_good = False

# Final summary
print("\n" + "="*80)
if all_good:
    print("SUCCESS! ALL STEPS COMPLETED")
    print("="*80)
    print("\n✅ Repository cleaned")
    print("✅ Fresh data generated (1000 patients)")
    print("✅ Features engineered (9 causal features)")
    print("✅ Models trained (3 algorithms)")
    print("\n📌 Next Steps:")
    print("   1. View experiments: mlflow ui")
    print("   2. Browse at: http://localhost:5000")
    print("   3. Start API: python api/main.py")
    print("   4. Test API at: http://localhost:8000/docs")
else:
    print("COMPLETED WITH WARNINGS")
    print("="*80)
    print("\nSome artifacts are missing. Check output above.")

print("\n" + "="*80 + "\n")
