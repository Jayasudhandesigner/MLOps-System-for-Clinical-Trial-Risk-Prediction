"""
api/test_api.py
===============
Quick API test script

Tests the prediction endpoint with example data.
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

# Test cases
test_cases = [
    {
        "name": "High-Risk Patient",
        "data": {
            "patient_id": "P-TEST-001",
            "age": 72,
            "gender": "Female",
            "treatment_group": "Placebo",
            "trial_phase": "Phase III",
            "days_in_trial": 150,
            "visits_completed": 2,
            "last_visit_day": 90,
            "adverse_events": 8
        }
    },
    {
        "name": "Low-Risk Patient",
        "data": {
            "patient_id": "P-TEST-002",
            "age": 45,
            "gender": "Male",
            "treatment_group": "Active",
            "trial_phase": "Phase I",
            "days_in_trial": 60,
            "visits_completed": 2,
            "last_visit_day": 55,
            "adverse_events": 1
        }
    }
]

def test_health():
    """Test health check endpoint"""
    print("=" * 80)
    print("Testing /health endpoint...")
    print("=" * 80)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predictions():
    """Test prediction endpoint"""
    print("=" * 80)
    print("Testing /predict endpoint...")
    print("=" * 80)
    
    for test_case in test_cases:
        print(f"\n📊 {test_case['name']}:")
        print(f"Input: {json.dumps(test_case['data'], indent=2)}")
        
        response = requests.post(
            f"{API_URL}/predict",
            json=test_case['data']
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prediction:")
            print(f"   Patient ID: {result['patient_id']}")
            print(f"   Dropout Prediction: {result['dropout_prediction']}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Recommended Action: {result['recommended_action']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
        
        print("-" * 80)

def test_stats():
    """Test stats endpoint"""
    print("\n" + "=" * 80)
    print("Testing /stats endpoint...")
    print("=" * 80)
    
    response = requests.get(f"{API_URL}/stats")
    if response.status_code == 200:
        stats = response.json()
        print(json.dumps(stats, indent=2))
    else:
        print(f"❌ Error: {response.status_code}")
    print()

if __name__ == "__main__":
    print("\n🧪 API TEST SUITE\n")
    
    try:
        # Test endpoints
        test_health()
        test_predictions()
        test_stats()
        
        print("=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API")
        print("   Make sure the API is running: python api/main.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")
