# 🛡️ DEPLOYMENT & SECURITY GUIDE (DAY 18)

## 📌 Overview
This system is hardened for production use in healthcare environments. It follows a "Zero Trust" architecture where the API Gateway (FastAPI) acts as the secure boundary between the frontend (React) and the ML infrastructure.

## 🔐 Security Audit
| Feature | Implementation | Status |
| :--- | :--- | :--- |
| **Authentication** | API Key (Header: `x-api-key`) | ✅ Enforced |
| **CORS** | Configured for React clients | ✅ Active |
| **Secrets** | Kubernetes Secrets (Env Injection) | ✅ Active |
| **PII Protection** | `patient_id` hashed in logs | ✅ Active |
| **Resource Limits** | CPU/Memory limits defined | ✅ Active |

---

## 🚀 Deployment Steps (Kubernetes)

### 1. Create ConfigMap (Policy)
Stores non-sensitive configuration (thresholds, model stages).
```bash
kubectl apply -f k8s/configmap.yaml
```

### 2. Create Secret (Capabilities)
Stores sensitive credentials. **DO NOT commit actual keys to Git.**
```bash
# Generate a secure key (e.g., using openssl)
KEY=$(openssl rand -base64 32)
echo "Generated Key: $KEY"

# Create Secret directly
kubectl create secret generic dropout-secrets \
  --from-literal=API_KEY="$KEY"
```

### 3. Deploy Application
Deploys the secured API with autoscaling and load balancing.
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 4. Verify Access
To access the API, you must provide the API Key header.

**Test with curl:**
```bash
# Get LoadBalancer IP
IP=$(kubectl get svc dropout-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# ❌ Fail (401 Unauthorized)
curl -X POST http://$IP/predict -d '...'

# ✅ Success (200 OK)
curl -X POST http://$IP/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: $KEY" \
  -d '{"patient_id": "P-001", "age": 65, ...}'
```

---

## 💰 Cost Optimization
- **Horizontal Pod Autoscaling (HPA)**: Scales from **2 to 10 replicas** based on CPU load (Target: 70%).
- **Resource Limits**:
  - Request: 0.25 vCPU, 512Mi RAM (Guaranteed)
  - Limit: 0.50 vCPU, 1Gi RAM (Burst cap)
- **Retraining**: Only triggers on **verified drift**, preventing wasted compute.

## ⚛️ Frontend Integration (React)
The API is **React-Ready**.

**Environment Setup (`.env`):**
```bash
REACT_APP_API_URL=http://<LOAD_BALANCER_IP>
REACT_APP_API_KEY=<YOUR_SECURE_KEY>
```

**Fetch Example:**
```javascript
const response = await fetch(`${process.env.REACT_APP_API_URL}/predict`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': process.env.REACT_APP_API_KEY
  },
  body: JSON.stringify(data)
});
```
