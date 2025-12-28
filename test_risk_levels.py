"""
test_risk_levels.py
===================
Test API with patients ranging from VERY LOW to VERY HIGH dropout risk
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

# Test patients with varying risk levels
test_patients = [
    {
        "name": "VERY LOW RISK",
        "description": "Young, Active treatment, Phase I, excellent compliance",
        "data": {
            "patient_id": "P-LOW-001",
            "age": 30,
            "gender": "Male",
            "treatment_group": "Active",
            "trial_phase": "Phase I",
            "days_in_trial": 60,
            "visits_completed": 3,
            "last_visit_day": 58,
            "adverse_events": 0
        }
    },
    {
        "name": "LOW RISK",
        "description": "Middle-aged, Active treatment, good compliance",
        "data": {
            "patient_id": "P-LOW-002",
            "age": 45,
            "gender": "Female",
            "treatment_group": "Active",
            "trial_phase": "Phase II",
            "days_in_trial": 90,
            "visits_completed": 3,
            "last_visit_day": 85,
            "adverse_events": 1
        }
    },
    {
        "name": "MODERATE RISK",
        "description": "Senior, Control group, moderate compliance",
        "data": {
            "patient_id": "P-MOD-001",
            "age": 60,
            "gender": "Female",
            "treatment_group": "Control",
            "trial_phase": "Phase II",
            "days_in_trial": 120,
            "visits_completed": 3,
            "last_visit_day": 90,
            "adverse_events": 3
        }
    },
    {
        "name": "HIGH RISK",
        "description": "Senior, Placebo, Phase III, poor compliance",
        "data": {
            "patient_id": "P-HIGH-001",
            "age": 70,
            "gender": "Female",
            "treatment_group": "Placebo",
            "trial_phase": "Phase III",
            "days_in_trial": 150,
            "visits_completed": 2,
            "last_visit_day": 90,
            "adverse_events": 6
        }
    },
    {
        "name": "VERY HIGH RISK",
        "description": "Elderly, Placebo, Phase III, very poor compliance, many adverse events",
        "data": {
            "patient_id": "P-VERYHIGH-001",
            "age": 75,
            "gender": "Female",
            "treatment_group": "Placebo",
            "trial_phase": "Phase III",
            "days_in_trial": 180,
            "visits_completed": 1,
            "last_visit_day": 60,
            "adverse_events": 10
        }
    }
]

def test_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict(patient_data):
    """Make prediction request"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("\n" + "="*80)
    print("DROPOUT RISK PREDICTION TEST")
    print("Testing patients from VERY LOW to VERY HIGH risk")
    print("="*80 + "\n")
    
    # Check if API is running
    print("🔍 Checking if API is running...")
    if not test_health():
        print("\n❌ ERROR: API is not running!")
        print("\nPlease start the API server first:")
        print("   python api/main.py")
        print("\nThen run this test again.")
        return
    
    print("✅ API is online!\n")
    print("="*80 + "\n")
    
    # Test each patient
    for i, patient in enumerate(test_patients, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/5: {patient['name']}")
        print(f"{'='*80}")
        print(f"\n📋 Profile: {patient['description']}")
        print(f"\n📊 Patient Data:")
        print(f"   Age: {patient['data']['age']}")
        print(f"   Gender: {patient['data']['gender']}")
        print(f"   Treatment: {patient['data']['treatment_group']}")
        print(f"   Trial Phase: {patient['data']['trial_phase']}")
        print(f"   Days in Trial: {patient['data']['days_in_trial']}")
        print(f"   Visits Completed: {patient['data']['visits_completed']}")
        print(f"   Last Visit Day: {patient['data']['last_visit_day']}")
        print(f"   Adverse Events: {patient['data']['adverse_events']}")
        
        # Make prediction
        result = predict(patient['data'])
        
        if "error" in result:
            print(f"\n❌ ERROR: {result['error']}")
        else:
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"   Dropout Prediction: {result['dropout_prediction']} ({'WILL DROP OUT' if result['dropout_prediction'] == 1 else 'WILL STAY'})")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Recommended Action: {result['recommended_action']}")
        
        print("\n" + "-"*80)
        time.sleep(0.5)  # Small delay between requests
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
