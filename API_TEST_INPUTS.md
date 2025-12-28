# API Test JSON Inputs
## Clinical Trial Dropout Prediction API

**Backend URL**: http://localhost:8000  
**API Docs**: http://localhost:8000/docs  

---

## 🧪 TEST CASES - JSON Inputs

### 1️⃣ VERY LOW RISK Patient
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

**Expected Result**: Dropout = 0, Risk = Low, Action = standard_protocol

---

### 2️⃣ LOW RISK Patient
```json
{
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
```

**Expected Result**: Dropout = 0, Risk = Low, Action = standard_protocol

---

### 3️⃣ MODERATE RISK Patient
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

**Expected Result**: Dropout = 0, Risk = Moderate, Action = biweekly_check

---

### 4️⃣ HIGH RISK Patient
```json
{
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
```

**Expected Result**: Dropout = 1, Risk = High, Action = weekly_monitoring

---

### 5️⃣ VERY HIGH RISK (CRITICAL) Patient
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

**Expected Result**: Dropout = 1, Risk = Critical, Action = immediate_intervention

---

## 🔧 HOW TO TEST

### Option 1: Using curl (Command Line)

**Test Very Low Risk Patient**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"patient_id\":\"P-LOW-001\",\"age\":30,\"gender\":\"Male\",\"treatment_group\":\"Active\",\"trial_phase\":\"Phase I\",\"days_in_trial\":60,\"visits_completed\":3,\"last_visit_day\":58,\"adverse_events\":0}"
```

**Test Critical Risk Patient**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"patient_id\":\"P-VERYHIGH-001\",\"age\":75,\"gender\":\"Female\",\"treatment_group\":\"Placebo\",\"trial_phase\":\"Phase III\",\"days_in_trial\":180,\"visits_completed\":1,\"last_visit_day\":60,\"adverse_events\":10}"
```

---

### Option 2: Using PowerShell (Windows)

**Test Very Low Risk Patient**:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{"patient_id":"P-LOW-001","age":30,"gender":"Male","treatment_group":"Active","trial_phase":"Phase I","days_in_trial":60,"visits_completed":3,"last_visit_day":58,"adverse_events":0}'
```

**Test Critical Risk Patient**:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{"patient_id":"P-VERYHIGH-001","age":75,"gender":"Female","treatment_group":"Placebo","trial_phase":"Phase III","days_in_trial":180,"visits_completed":1,"last_visit_day":60,"adverse_events":10}'
```

---

### Option 3: Using Swagger UI (Interactive)

1. Open your browser
2. Go to: **http://localhost:8000/docs**
3. Click on **POST /predict**
4. Click **"Try it out"**
5. Paste any JSON from above into the Request body
6. Click **"Execute"**
7. See the response!

---

### Option 4: Using Postman

1. Open Postman
2. Create new **POST** request
3. URL: `http://localhost:8000/predict`
4. Headers: `Content-Type: application/json`
5. Body (raw JSON): Paste any JSON from above
6. Click **Send**

---

## 📊 EXPECTED RESPONSE FORMAT

```json
{
  "patient_id": "P-LOW-001",
  "dropout_prediction": 0,
  "risk_level": "Low",
  "recommended_action": "standard_protocol"
}
```

**Fields**:
- `dropout_prediction`: 0 (will stay) or 1 (will drop out)
- `risk_level`: "Low", "Moderate", "High", or "Critical"
- `recommended_action`: What clinicians should do

---

## 🔍 OTHER ENDPOINTS

### Health Check
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

### API Root
```bash
curl http://localhost:8000/
```

**Response**:
```json
{
  "message": "Clinical Trial Dropout Prediction API",
  "version": "1.0.0",
  "status": "healthy"
}
```

### Session Statistics
```bash
curl http://localhost:8000/stats
```

**Response**: Returns prediction statistics for current session

---

## ✅ BACKEND STATUS

- **URL**: http://localhost:8000
- **Status**: ✅ Running
- **Interactive Docs**: http://localhost:8000/docs
- **Model**: XGBoost/LightGBM (v3_causal)
- **Threshold**: 0.30 (balanced)

---

**Ready to test!** Use any of the JSON inputs above with your preferred method.
