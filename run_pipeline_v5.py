"""
run_pipeline_v5.py
==================
COMPLETE END-TO-END PIPELINE - V5 (Realistic Data)

Runs the full MLOps pipeline:
1. Generate realistic synthetic data
2. Ingest and validate
3. Feature engineering
4. Preprocessing
5. Model training (LightGBM with threshold 0.30)
6. MLflow tracking
7. DVC versioning
8. Git tagging

Run with: python run_pipeline_v5.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*70}")
    print(f"🔄 {description}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    print(f"✅ {description} - COMPLETE")
    return result

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  MLOPS PIPELINE V5 - REALISTIC DATA                  ║
║                                                                       ║
║  This pipeline creates production-ready models with:                 ║
║  - Realistic human behavior patterns                                 ║
║  - Threshold 0.30 (balanced)                                        ║
║  - Full ML

Ops tracking (MLflow + DVC)                         ║
║  - Git tagging for reproducibility                                   ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Step 1: Generate data
run_command(
    "python data/synthetic_data_realistic_v5.py",
    "Step 1: Generate Realistic Synthetic Data"
)

# Step 2: DVC add data
run_command(
    "dvc add data/raw/clinical_trials_realistic_v5.csv",
    "Step 2: Version Data with DVC"
)

# Step 3: Run pipeline
run_command(
    "python pipelines/local_pipeline.py",
    "Step 3: Run Training Pipeline"
)

# Step 4: Git commit
run_command(
    'git add .',
    "Step 4a: Stage Changes"
)

run_command(
    'git commit -m "feat: realistic data v5 with human behavior archetypes and threshold 0.30"',
    "Step 4b: Commit Changes"
)

# Step 5: Git tag
run_command(
    'git tag -a v5.0-realistic-data -m "V5: Realistic human behavior data, threshold 0.30, production-ready"',
    "Step 5: Create Git Tag"
)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ✅ PIPELINE COMPLETE!                            ║
╚══════════════════════════════════════════════════════════════════════╝

Next steps:
1. Push to GitHub:
   git push origin main
   git push origin v5.0-realistic-data

2. Push DVC data:
   dvc push

3. View results:
   mlflow ui

4. Test API:
   python api/main.py
   python api/test_api.py
""")
