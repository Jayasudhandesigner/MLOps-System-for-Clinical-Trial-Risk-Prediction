# Model Card — Clinical Trial Dropout Prediction

## Model Purpose
Predict likelihood of patient dropout during a clinical trial.

## Models Trained
- Logistic Regression (baseline)
- Random Forest Classifier

## Training Data
- Source: Internal clinical trial dataset
- Size: Small sample (demo)

## Features
- Demographics
- Trial participation metrics
- Adverse events

## Metrics
- ROC-AUC
- Recall (dropout class)

## Ethical Considerations
- Predictions should support intervention, not exclusion
- Model must not be used as sole decision-maker
