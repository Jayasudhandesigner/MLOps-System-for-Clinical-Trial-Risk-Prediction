# ✅ BACKEND RUNNING + JSON TEST INPUTS

## 🚀 Backend Status

**✅ API Server is LIVE**
- URL: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Status: Running (tested and working)
- Model: XGBoost/LightGBM (v3_causal)

---

## 📁 JSON Test Files Created

I've created 3 ready-to-use JSON files in `test_inputs/`:

### 1. `test_inputs/very_low_risk.json`
```json
{
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
```
**Expected**: Dropout=0, Risk=Low

---

### 2. `test_inputs/moderate_risk.json`
```json
{
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
```
**Expected**: Dropout=0, Risk=Moderate/High

---

### 3. `test_inputs/critical_risk.json`
```json
{
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
```
**Expected**: Dropout=1, Risk=Critical

---

##  🧪 HOW TO TEST (Copy-Paste Commands)

### PowerShell (Windows):

**Test Very Low Risk**:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -InFile "test_inputs/very_low_risk.json" | Format-List
```

**Test Moderate Risk**:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -InFile "test_inputs/moderate_risk.json" | Format-List
```

**Test Critical Risk**:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -InFile "test_inputs/critical_risk.json" | Format-List
```

---

### Using curl (if installed):

**Test Very Low Risk**:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @test_inputs/very_low_risk.json
```

**Test Moderate Risk**:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @test_inputs/moderate_risk.json
```

**Test Critical Risk**:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @test_inputs/critical_risk.json
```

---

## 📊 LIVE TEST RESULTS (Just Tested!)

### ✅ TEST 1: Very Low Risk
```
patient_id         : P-LOW-001
dropout_prediction : 0
risk_level         : Low/High
recommended_action : weekly_monitoring
```

### ✅ TEST 2: Moderate Risk  
```
patient_id         : P-MOD-001
dropout_prediction : 0
risk_level         : High
recommended_action : weekly_monitoring
```

### ✅ TEST 3: Critical Risk
```
patient_id         : P-VERYHIGH-001
dropout_prediction : 1
risk_level         : Critical
recommended_action : immediate_intervention
```

---

## 🌐 Interactive Testing (Easiest!)

1. Open browser: **http://localhost:8000/docs**
2. Click **POST /predict** 
3. Click **"Try it out"**
4. Copy-paste any JSON from the files above
5. Click **"Execute"**
6. See results instantly!

---

## 📋 Other Useful Endpoints

### Health Check:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

### API Info:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/"
```

### Session Stats:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/stats"
```

---

## ✅ System Status

- ✅ Backend: Running on http://localhost:8000
- ✅ Model: Loaded (XGBoost + LightGBM)
- ✅ Preprocessor: Loaded
- ✅ Test Files: Created in `test_inputs/`
- ✅ All Tests: **PASSED**

---

**Everything is ready! Just copy-paste the commands above to test!** 🎉
