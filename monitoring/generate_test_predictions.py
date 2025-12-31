"""
Generate synthetic predictions for drift detection testing.
This simulates live predictions without needing the API running.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_test_predictions(n: int = 50):
    """Generate N synthetic predictions for testing drift detection."""
    
    output_path = Path("monitoring/live_predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {n} synthetic predictions...")
    
    with open(output_path, "w") as f:
        base_time = datetime.utcnow() - timedelta(hours=n)
        
        for i in range(n):
            # Simulate input data
            input_data = {
                "patient_id": f"P-SIM-{str(i+1).zfill(4)}",
                "age": random.randint(25, 85),
                "gender": random.choice(["Male", "Female"]),
                "treatment_group": random.choice(["Active", "Control", "Placebo"]),
                "trial_phase": random.choice(["Phase I", "Phase II", "Phase III"]),
                "days_in_trial": random.randint(30, 180),
                "visits_completed": random.randint(1, 6),
                "last_visit_day": random.randint(20, 150),
                "adverse_events": random.randint(0, 8)
            }
            
            # Ensure last_visit_day <= days_in_trial
            input_data["last_visit_day"] = min(
                input_data["last_visit_day"], 
                input_data["days_in_trial"]
            )
            
            # Simulate output (biased toward realistic risk assessment)
            prob = random.uniform(0.1, 0.9)
            
            output_data = {
                "dropout_prediction": 1 if prob >= 0.5 else 0,
                "dropout_probability": round(prob, 4),
                "risk_level": "Critical" if prob >= 0.8 else ("Medium" if prob >= 0.4 else "Low"),
                "recommended_action": "retention_team_deployment" if prob >= 0.8 else (
                    "nurse_doctor_consultation" if prob >= 0.4 else "automated_sms_alert"
                ),
                "intervention_cost": 500.0 if prob >= 0.8 else (45.0 if prob >= 0.4 else 0.5)
            }
            
            record = {
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "input": input_data,
                "output": output_data
            }
            
            f.write(json.dumps(record) + "\n")
    
    print(f"✅ Generated {n} predictions to {output_path}")
    return n


if __name__ == "__main__":
    generate_test_predictions(50)
