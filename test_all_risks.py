"""
Test predictions for all risk categories via command line
"""
import requests
import json

# API endpoint
API_URL = "http://localhost:8000/predict"

# Test cases for different risk levels
test_cases = [
    {
        "name": "VERY LOW RISK",
        "description": "Young, Active treatment, Phase I, good compliance",
        "data": {
            "patient_id": "P-VERYLOW-001",
            "age": 30,
            "gender": "Male",
            "treatment_group": "Active",
            "trial_phase": "Phase I",
            "days_in_trial": 45,
            "visits_completed": 2,
            "last_visit_day": 40,
            "adverse_events": 0
        }
    },
    {
        "name": "LOW RISK",
        "description": "Middle-aged, Active treatment, Phase I",
        "data": {
            "patient_id": "P-LOW-002",
            "age": 45,
            "gender": "Female",
            "treatment_group": "Active",
            "trial_phase": "Phase I",
            "days_in_trial": 60,
            "visits_completed": 2,
            "last_visit_day": 55,
            "adverse_events": 1
        }
    },
    {
        "name": "MEDIUM RISK",
        "description": "Older adult, Control, Phase II, moderate compliance",
        "data": {
            "patient_id": "P-MEDIUM-003",
            "age": 58,
            "gender": "Male",
            "treatment_group": "Control",
            "trial_phase": "Phase II",
            "days_in_trial": 100,
            "visits_completed": 3,
            "last_visit_day": 80,
            "adverse_events": 4
        }
    },
    {
        "name": "HIGH RISK",
        "description": "Senior, Placebo, Phase III, poor compliance",
        "data": {
            "patient_id": "P-HIGH-004",
            "age": 68,
            "gender": "Female",
            "treatment_group": "Placebo",
            "trial_phase": "Phase III",
            "days_in_trial": 120,
            "visits_completed": 2,
            "last_visit_day": 75,
            "adverse_events": 7
        }
    },
    {
        "name": "VERY HIGH RISK",
        "description": "Elderly, Placebo, Phase III, very poor compliance",
        "data": {
            "patient_id": "P-VERYHIGH-005",
            "age": 75,
            "gender": "Male",
            "treatment_group": "Placebo",
            "trial_phase": "Phase III",
            "days_in_trial": 150,
            "visits_completed": 1,
            "last_visit_day": 60,
            "adverse_events": 12
        }
    }
]

print("="*80)
print("DROPOUT PREDICTION TESTING - ALL RISK CATEGORIES")
print("="*80)
print(f"API Endpoint: {API_URL}")
print(f"Threshold: 0.30")
print("="*80)
print()

results = []

for test in test_cases:
    print(f"{'─'*80}")
    print(f"TEST: {test['name']}")
    print(f"{'─'*80}")
    print(f"Description: {test['description']}")
    print()
    print("Patient Profile:")
    print(f"  Age: {test['data']['age']}")
    print(f"  Treatment: {test['data']['treatment_group']}")
    print(f"  Phase: {test['data']['trial_phase']}")
    print(f"  Days in trial: {test['data']['days_in_trial']}")
    print(f"  Visits: {test['data']['visits_completed']}")
    print(f"  Adverse events: {test['data']['adverse_events']}")
    print()
    
    try:
        response = requests.post(API_URL, json=test['data'], timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            
            # Determine if prediction matches expected risk
            expected_dropout = 1 if "HIGH" in test['name'] else 0
            actual_dropout = result['dropout_prediction']
            match = "✅" if expected_dropout == actual_dropout else "⚠️"
            
            print("PREDICTION RESULT:")
            print(f"  {match} Dropout Prediction: {actual_dropout} ({'DROPOUT' if actual_dropout == 1 else 'RETAIN'})")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Recommended Action: {result['recommended_action']}")
            
            results.append({
                'category': test['name'],
                'prediction': actual_dropout,
                'risk_level': result['risk_level'],
                'match': match
            })
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API")
        print("   Make sure the API is running: python api/main.py")
        break
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print()

# Summary
if results:
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"{'Category':<20} {'Prediction':<15} {'Risk Level':<15} {'Match'}")
    print("-"*80)
    for r in results:
        pred_text = "DROPOUT" if r['prediction'] == 1 else "RETAIN"
        print(f"{r['category']:<20} {pred_text:<15} {r['risk_level']:<15} {r['match']}")
    print()
    print("="*80)
    print(f"Total tests: {len(results)}")
    matches = sum(1 for r in results if r['match'] == '✅')
    print(f"Expected matches: {matches}/{len(results)}")
    print("="*80)
