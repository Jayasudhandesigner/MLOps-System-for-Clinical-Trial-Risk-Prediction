import requests
import json

# Test different risk profiles with threshold 0.35
test_cases = [
    {
        'name': 'Very High Risk',
        'data': {
            'patient_id': 'P-VERY-HIGH',
            'age': 72,
            'gender': 'Female',
            'treatment_group': 'Placebo',
            'trial_phase': 'Phase III',
            'days_in_trial': 150,
            'visits_completed': 1,
            'last_visit_day': 60,
            'adverse_events': 10
        }
    },
    {
        'name': 'High Risk',
        'data': {
            'patient_id': 'P-HIGH',
            'age': 68,
            'gender': 'Male',
            'treatment_group': 'Placebo',
            'trial_phase': 'Phase III',
            'days_in_trial': 120,
            'visits_completed': 2,
            'last_visit_day': 80,
            'adverse_events': 6
        }
    },
    {
        'name': 'Medium Risk',
        'data': {
            'patient_id': 'P-MEDIUM',
            'age': 55,
            'gender': 'Female',
            'treatment_group': 'Control',
            'trial_phase': 'Phase II',
            'days_in_trial': 90,
            'visits_completed': 3,
            'last_visit_day': 75,
            'adverse_events': 3
        }
    },
    {
        'name': 'Low Risk',
        'data': {
            'patient_id': 'P-LOW',
            'age': 45,
            'gender': 'Male',
            'treatment_group': 'Active',
            'trial_phase': 'Phase I',
            'days_in_trial': 60,
            'visits_completed': 2,
            'last_visit_day': 55,
            'adverse_events': 1
        }
    },
    {
        'name': 'Very Low Risk',
        'data': {
            'patient_id': 'P-VERY-LOW',
            'age': 35,
            'gender': 'Female',
            'treatment_group': 'Active',
            'trial_phase': 'Phase I',
            'days_in_trial': 45,
            'visits_completed': 2,
            'last_visit_day': 40,
            'adverse_events': 0
        }
    }
]

print('=' * 80)
print('PREDICTION TESTS WITH THRESHOLD 0.35')
print('=' * 80)
print()

for test in test_cases:
    try:
        r = requests.post('http://localhost:8000/predict', json=test['data'])
        result = r.json()
        pred_text = "DROPOUT" if result["dropout_prediction"] == 1 else "RETAIN"
        
        print(f"{test['name']}:")
        print(f"  Prediction: {result['dropout_prediction']} ({pred_text})")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Action: {result['recommended_action']}")
        print()
    except Exception as e:
        print(f"{test['name']}: ERROR - {e}")
        print()

# Summary
print('=' * 80)
print('SUMMARY')
print('=' * 80)
