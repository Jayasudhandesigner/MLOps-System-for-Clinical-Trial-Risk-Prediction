# Feature Specification

## Numerical
- age
- visits_completed
- adverse_events
- days_in_trial

## Categorical (One-Hot Encoded)
- gender
- treatment_group
- trial_phase

## Target
- dropout (binary)

## Rules
- No feature uses future information
- Encoding fitted on training data only
- Same logic used in training & inference