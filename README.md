# Health Insurance Fraud Intelligence Platform

A full-stack, multi-approach fraud detection system for insurance claims, integrating **Classical Machine Learning (Approach 1)**, **Deep Learning (Approach 2)**, and an **Agentic AI Pipeline (Approach 3)** into a unified web dashboard.

---

## Architecture Overview

The system runs as three connected services:

```text
React Frontend (Vite + TypeScript)
        ↓  (Port 5173 — proxies /api and /agentic)
Node.js Backend Gateway (Express + TypeScript)
        ↓  (Port 3000 — proxies /predict/dl and /predict/ml)
Python ML Inference Service (FastAPI + PyTorch + scikit-learn)
        ↓  (Port 8000)
[5 Classical ML Models] + [5 Deep Learning Models] + [Agentic Audit Trail]
```

### Supported Models

- **Classical ML (Approach 1):** `RandomForest`, `XGBoost`, `GradientBoosting`, `LogisticRegression`, `SVM`
- **Deep Learning (Approach 2):**
  - Supervised: `MLP` (Baseline), `ResidualMLP`, `CNN1D`, `TabTransformer`
  - Unsupervised: `Autoencoder` (Reconstruction anomaly detector with fitted 95th percentile threshold = 0.1286)
- **Agentic AI (Approach 3):** 4-stage pipeline (`ingestion` → `validation` → `risk-scorer` → `xai-explainer`)

---

## Prerequisites

1. **Python Virtual Environment**:
   - Location: `Approch-2-DL/.venv`
   - Dependencies: `torch`, `scikit-learn`, `xgboost`, `fastapi`, `uvicorn`, `joblib`, `pandas`, `numpy`
2. **Node.js**:
   - Version 18+ and `npm`

---

## How to Run the Entire Pipeline

To run the complete system, open **three separate terminal windows** from the repository root (`E:\mini-project`):

### Terminal 1: Python FastAPI ML Service (Port 8000)

```powershell
# Navigate to project root
cd E:\mini-project

# Start FastAPI service using the Approach-2 virtual environment
.\Approch-2-DL\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir ml-service
```

- **Health check:** `http://127.0.0.1:8000/health`
- **Interactive API Docs (Swagger):** `http://127.0.0.1:8000/docs`

---

### Terminal 2: Node.js Backend Gateway (Port 3000)

```powershell
# Navigate to backend directory
cd E:\mini-project\backend\node-api

# Install dependencies (first time only)
npm install

# Start the Node.js API server
npm run dev
```

- **Health check:** `http://127.0.0.1:3000/api/health`
- **Metrics check:** `http://127.0.0.1:3000/api/results/dl/all`

---

### Terminal 3: React Frontend Application (Port 5173)

```powershell
# Navigate to frontend directory
cd E:\mini-project\frontend

# Install dependencies (first time only)
npm install

# Start Vite dev server
npm run dev
```

- **Web Application URL:** Open `http://127.0.0.1:5173` in your browser.

---

## Quick Verification & Smoke Test

Once all three services are running, run this PowerShell script to verify all endpoints and proxy chains:

```powershell
$body = @{
    ClaimAmount = 5000.0
    ClaimDate = "2024-06-15"
    PatientAge = 45
    PatientGender = "F"
    ProviderSpecialty = "Cardiology"
    PatientIncome = 60000.0
    PatientMaritalStatus = "Married"
    PatientEmploymentStatus = "Employed"
    ClaimType = "Routine"
    ClaimSubmissionMethod = "Online"
    Cluster = 1
} | ConvertTo-Json

# 1. Test Classical ML (RandomForest)
Invoke-RestMethod "http://127.0.0.1:3000/api/ml/predict?model=RandomForest" -Method POST -Body $body -ContentType "application/json" | Select-Object model_name, prediction, fraud_score

# 2. Test Deep Learning (TabTransformer)
Invoke-RestMethod "http://127.0.0.1:3000/api/dl/predict?model=TabTransformer" -Method POST -Body $body -ContentType "application/json" | Select-Object model_name, prediction, fraud_score

# 3. Test Deep Learning (Autoencoder)
Invoke-RestMethod "http://127.0.0.1:3000/api/dl/predict?model=Autoencoder" -Method POST -Body $body -ContentType "application/json" | Select-Object model_name, prediction, fraud_score

# 4. Test Agentic AI
$agenticBody = @{
    claim_amount = 5000.0
    patient_age = 45
    patient_income = 60000.0
    claim_type = "Routine"
    provider_specialty = "Cardiology"
    cluster = 1
    submission_method = "Online"
} | ConvertTo-Json
Invoke-RestMethod "http://127.0.0.1:3000/agentic/predict" -Method POST -Body $agenticBody -ContentType "application/json" | Select-Object prediction, fraud_score, model_name
```

---

## API Summary

| Method | Path | Target Service | Description |
|---|---|---|---|
| `GET` | `/health` | FastAPI (`:8000`) | FastAPI service health check |
| `POST` | `/predict/dl?model=<name>` | FastAPI (`:8000`) | Direct DL inference (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`, `Autoencoder`) |
| `POST` | `/predict/ml?model=<name>` | FastAPI (`:8000`) | Direct Classical ML inference (`RandomForest`, `XGBoost`, `SVM`, etc.) |
| `GET` | `/api/health` | Node.js (`:3000`) | Node gateway health check |
| `POST` | `/api/dl/predict?model=<name>` | Node.js (`:3000`) | Proxies to FastAPI DL inference |
| `POST` | `/api/ml/predict?model=<name>` | Node.js (`:3000`) | Proxies to FastAPI Classical ML inference |
| `GET` | `/api/results/ml` | Node.js (`:3000`) | Classical ML evaluation benchmarks (`model_results.json`) |
| `GET` | `/api/results/dl/all` | Node.js (`:3000`) | Consolidated benchmark metrics across all 5 DL architectures |
| `POST` | `/agentic/predict` | Node.js (`:3000`) | Multi-agent risk analysis & audit trail |

---

## Production Build Verification

To verify that the frontend and backend compile without errors:

```powershell
# Frontend TypeScript check and production bundle build
cd E:\mini-project\frontend
npm run build

# Backend TypeScript check and build
cd E:\mini-project\backend\node-api
npm run build
```
