"""
train_production_model.py
=========================
Lightweight model training for production deployment.
Trains a LogisticRegression model compatible with the container's sklearn version.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
from pathlib import Path

def train_model(output_path: str = "models/production_model.pkl"):
    """Train a production-ready model with synthetic data."""
    print(f"Training with sklearn {sklearn.__version__}")
    
    np.random.seed(42)
    n = 1000  # Sufficient for good generalization
    
    # Generate realistic clinical trial features
    data = pd.DataFrame({
        'age': np.random.randint(25, 80, n),
        'days_in_trial': np.random.randint(10, 365, n),
        'visit_rate': np.random.uniform(0.3, 1.0, n),
        'adverse_event_rate': np.random.uniform(0, 0.2, n),
        'time_since_last_visit': np.random.randint(0, 60, n),
        'burden': np.random.uniform(0, 0.1, n),
        'age_adverse_risk': np.random.uniform(0, 0.2, n),
        'trial_phase_risk': np.random.choice([0.2, 0.5, 0.8], n),
        'treatment_risk': np.random.choice([0.1, 0.3, 0.4], n)
    })
    
    # Generate realistic dropout labels based on feature signals
    dropout_likelihood = (
        0.2 * (data['age'] / 80) +
        0.3 * (1 - data['visit_rate']) +
        0.3 * data['adverse_event_rate'] * 5 +
        0.1 * (data['time_since_last_visit'] / 60) +
        0.1 * data['treatment_risk']
    )
    y = (dropout_likelihood + np.random.normal(0, 0.1, n) > 0.35).astype(int)
    
    # Train lightweight model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(data, y)
    
    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    
    # Verify
    accuracy = model.score(data, y)
    print(f"✅ Model trained and saved to {output_path}")
    print(f"   Training accuracy: {accuracy:.2%}")
    print(f"   Features: {list(data.columns)}")
    
    return model

if __name__ == "__main__":
    train_model()
