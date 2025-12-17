import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
n_rows = 1000

data = {
    'patient_id': [f'P-{np.random.randint(1000, 9999)}' for _ in range(n_rows)],
    'age': np.random.randint(18, 85, size=n_rows),
    'gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_rows),
    'treatment_group': np.random.choice(['Active', 'Control', 'Placebo'], n_rows),
    'trial_phase': np.random.choice(['Phase I', 'Phase II', 'Phase III'], n_rows),
    'visits_completed': np.random.randint(0, 13, size=n_rows),
    'adverse_events': np.random.poisson(0.5, n_rows),
    'days_in_trial': np.random.randint(1, 365, size=n_rows),
    'dropout': np.random.choice([0, 1], n_rows, p=[0.8, 0.2])
}

df = pd.DataFrame(data)
df.to_csv('./raw/clinical_trials.csv', index=False)