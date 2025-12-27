# Experiments Directory

**Research & Development Sandbox**

This directory contains experimental code, one-off scripts, and prototypes.
Code here is **NOT** part of the production pipeline.

## Guidelines

1. **Isolation:** Experimental scripts should import from `src.core` but should not be imported BY `src.core`.
2. **Naming:** Use descriptive prefixes (e.g., `exp_01_feature_selection.py`, `proto_new_model.py`).
3. **Clean-up:** Successful experiments should be refactored into `src.core` and the script removed or archived.
4. **No Production Use:** Do not run these scripts in production workflows.

## Common Scripts

- `compare_models.py`: Ad-hoc model comparison
- `tune_hyperparams.py`: Grid search/Random search experiments
- `data_analysis.ipynb`: Exploratory data analysis (EDA)
