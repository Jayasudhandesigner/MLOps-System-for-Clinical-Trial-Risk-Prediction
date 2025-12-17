# Clinical Trial Data Contract

## Required Columns
- patient_id (string, non-null)
- age (int, non-null)
- gender (category, non-null)
- treatment_group (category, non-null)
- trial_phase (category, non-null)
- visits_completed (int, non-null)
- adverse_events (int, non-null)
- days_in_trial (int, non-null)
- dropout (int, {0,1})

## Assumptions
- Each row represents one patient
- Dropout is recorded at trial end
- No duplicate patient_id values
