# Clinical Trial Dropout Risk Prediction (MLOps)

## Problem Statement
Clinical trials suffer from high patient dropout rates, leading to increased cost,
extended timelines, and biased results. This system predicts the likelihood of
patient dropout during an ongoing trial so that intervention strategies can be applied.

## ML Task
Binary classification:
- 1 → Patient likely to drop out
- 0 → Patient likely to complete the trial

## Intended Users
Clinical operations and trial management teams.

## Key Metrics
- ROC-AUC (overall discrimination)
- Recall on dropout class (business-critical)

## System Goal
Build a reproducible, production-grade MLOps pipeline with CI/CD,
monitoring, and automated retraining.
