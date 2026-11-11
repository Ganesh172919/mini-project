# Full Project Documentation & Engineering Handbook

**Project:** Health Insurance Fraud Intelligence Platform (`Medical Insurance Claim Fraud Detection`)  
**Institution:** Department of Computer Science and Engineering, IIIT Dharwad  
**Authors:** B. Varshith (`23BDS011`), J. Ganesh Kumar Reddy (`23BDS024`), M. Jagadeeswar (`23BDS033`)  
**Faculty Guide:** Prof. Ramesh Athe  

---

## 1. Repository Map & Module Ownership

| Directory / File | Approach / Role | Primary Owner | Status & Key Contents |
|---|---|---|---|
| `Health Insurance Fraud Claims.xlsx` | Source Dataset (Frozen) | Shared | 4,500 claims × 19 cols (`6.00%` fraud, SHA-256 `952b6e2...45`). Never modified. |
| `Approch-1/` | Classical ML (`ML`) | J. Ganesh Kumar Reddy (`23BDS024`) | EDA, preprocessing (`29f`), training (`LR`, `RF`, `XGB`, `SVM`, `HGB`), saved models, evaluation JSONs/PNGs. |
| `Approch-2-DL/` | Deep Learning (`DL`) | M. Jagadeeswar (`23BDS033`) | `MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`, `Autoencoder` checkpoints (`PyTorch`) + metrics/smoke tests. |
| `Approach-3-Agentic-AI/` | Agentic & XAI Scaffold | B. Varshith (`23BDS011`) | Port `:8020` FastAPI prototype (`predict.py` deterministic formula, `XAIExplainer` SHA-256 + SHAP-approx helper). |
| `ml-service/` | Unified Inference Server (`:8000`) | M. Jagadeeswar (`23BDS033`) | Production FastAPI service exposing `/predict/classical/{model}` (5 models) and `/predict/dl/{model}` (5 models). |
| `backend/node-api/` | Express API Gateway (`:3000`) | M. Jagadeeswar & B. Varshith | TypeScript Express server: proxies ML/DL to `:8000`, serves `/agentic/predict` (`AgenticEnsemble-v1`) & Gemini `/api/chat`. |
| `frontend/` | Interactive Web App (`:5173`) | M. Jagadeeswar & B. Varshith | React 19 + Vite + TypeScript + Tailwind v4 UI: Home, Classical ML, DL, Agentic Workflow (`6-agent` cards) & Chatbox. |
| `submission/` | Controlled Research Benchmark & Browser Lab (`:8080`) | J. Ganesh Kumar Reddy (`23BDS024`) | 32-run pipeline (`8 models × 4 experiments`), bootstrap CI, exported browser weights (`<4.42e-7` parity), static site. |
| `docs/` | Project Documentation Suite | Unified Team Deliverable | Literature Survey, Problem Formulation, Innovation & Implementation, Full Documentation. |
| `deliverables/` | Reports, IEEE Paper & Slides | Unified Team Deliverable | IIIT Dharwad LaTeX Report + PDF, IEEE LaTeX + PDF + DOCX Paper, 20-Slide Presentation Deck + PDF, 4 Standalone Doc PDFs, High-DPI Figures. |
| `PROJECT_PROGRESS.md`, `DECISIONS.md` | Engineering Contracts & Logs | Unified Team | Canonical 29-feature vs. Track B boundaries, uncalibrated score semantics, and benchmark snapshots. |

---

## 2. Two Complementary Verification Tracks (`Track A` vs. `Track B`)

1. **Track A — Full-Stack 10-Model Contract (`Approch-1/`, `Approch-2-DL/`, `ml-service/`, `backend/`, `frontend/`):**
   - Evaluates all **5 Classical ML** models and all **5 Deep Learning** models on the canonical 29-feature baseline contract (`Train=3,150 [189]`, `Val=675 [41]`, `Test=675 [40]`, seed `42`, including `Cluster`).
   - Supervised classification cutoff reported at standard `0.50` (`Approch-2-DL/results/*_metrics.json` and `Approch-1/evaluation/model_results.json`), plus Autoencoder at validation 95th-percentile reconstruction MSE `0.128588`.
2. **Track B — Leakage-Governed Research Lab (`submission/`):**
   - Evaluates all **8 supervised models** (`LR`, `RF`, `XGB`, `SVM`, `HGB`, `MLP`, `ResidualMLP`, `CNN1D`) across **4 protocols** (`baseline 29f`, `intake 28f without Cluster`, `no_amount 27f without Cluster & ClaimAmount`, and `temporal intake >= 2024-03-11`) — **`32` runs / `128` val+test metric rows**.
   - Optimizes the decision threshold strictly on the validation set (`validation_f1` policy) and exports pure-JSON browser weights verified against Python.

---

## 3. Complete Verified Quantitative Results

### 3.1 Track A — Held-Out Test Benchmarks (29 Features, `N=675`, 40 Frauds)

| Model | Family | Test Acc (%) | Prec (%) | Rec (%) | F1 (%) | ROC-AUC (%) | PR-AUC / AP (%) | Confusion (`TN/FP/FN/TP`) | Cutoff |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| **XGBoost** | Classical ML | 99.85 | 97.56 | 100.00 | **98.77** | **100.00** | **100.00** | `634 / 1 / 0 / 40` | `0.50` |
| **RandomForest** | Classical ML | 99.85 | 100.00 | 97.50 | 98.73 | 99.996 | 99.94 | `635 / 0 / 1 / 39` | `0.50` |
| **HistGradientBoosting** | Classical ML | 99.85 | 100.00 | 97.50 | 98.73 | 98.47 | 97.65 | `635 / 0 / 1 / 39` | `0.50` |
| **TabTransformer** | Supervised DL | 98.67 | 81.63 | 100.00 | 89.89 | **99.94** | **99.11** | `626 / 9 / 0 / 40` | `0.50` |
| **CNN1D** | Supervised DL | 98.81 | 83.33 | 100.00 | **90.91** | 99.92 | 98.82 | `627 / 8 / 0 / 40` | `0.50` |
| **ResidualMLP** | Supervised DL | 98.52 | 80.00 | 100.00 | 88.89 | 99.75 | 95.61 | `625 / 10 / 0 / 40` | `0.50` |
| **MLP** | Supervised DL | 98.81 | 84.78 | 97.50 | 90.70 | 99.73 | 95.38 | `628 / 7 / 1 / 39` | `0.50` |
| **SVM (RBF)** | Classical ML | 97.78 | 74.51 | 95.00 | 83.52 | 99.55 | 92.52 | `622 / 13 / 2 / 38` | `0.50` |
| **LogisticRegression** | Classical ML | 96.44 | 62.50 | 100.00 | 76.92 | 99.50 | 91.86 | `611 / 24 / 0 / 40` | `0.50` |
| **Autoencoder** *(Neg. Result)* | Unsupervised DL | 90.37 | 3.70 | 2.50 | **2.99** | 46.02 | **5.23** | `609 / 26 / 39 / 1` | MSE `0.1286` |

### 3.2 Track B — Four-Experiment Sensitivity & Chronological Ablation (`validation_f1` Policy)

| Model | Baseline Test `AP / F1` (`29f`) | Intake Test `AP / F1` (`28f`, No Cluster) | No-Amount Test `AP / F1` (`27f`) | Temporal Intake Test `AP / F1` (`>=2024-03-11`) | Intake Errors (`FP / FN`) & $\tau^*$ |
|---|---:|---:|---:|---:|---|
| **RandomForest** *(Selected Default)* | `0.9994 / 0.9750` | **`1.0000 / 0.9873`** | `0.2066 / 0.3636` | **`1.0000 / 1.0000`** | `FP=0, FN=1`, $\tau^*=0.4167$ |
| **XGBoost** | `1.0000 / 0.9877` | `1.0000 / 0.9877` | `0.2128 / 0.3524` | `1.0000 / 1.0000` | `FP=1, FN=0`, $\tau^*=0.2121$ |
| **GradientBoosting (`HGB`)** | `0.9765 / 0.9873` | `0.9765 / 0.9873` | `0.2404 / 0.3636` | `1.0000 / 1.0000` | `FP=0, FN=1`, $\tau^*=1.0000$ |
| **CNN1D** | `0.9882 / 0.9268` | `0.9667 / 0.8767` | `0.2287 / 0.2788` | `0.9172 / 0.8108` | `FP=1, FN=8`, $\tau^*=0.9691$ |
| **ResidualMLP** | `0.9567 / 0.9286` | `0.8765 / 0.8315` | `0.2246 / 0.3438` | `0.7979 / 0.7229` | `FP=12, FN=3`, $\tau^*=0.2751$ |
| **MLP** | `0.9538 / 0.9136` | `0.8866 / 0.8222` | `0.2303 / 0.3077` | `0.8581 / 0.6957` | `FP=13, FN=3`, $\tau^*=0.8359$ |
| **LogisticRegression** | `0.9186 / 0.7835` | `0.8525 / 0.7327` | `0.2318 / 0.3191` | `0.8799 / 0.7397` | `FP=24, FN=3`, $\tau^*=0.7618$ |
| **SVM (RBF)** | `0.9252 / 0.8974` | `0.8181 / 0.7711` | `0.1869 / 0.3041` | `0.8396 / 0.7619` | `FP=11, FN=8`, $\tau^*=0.5859$ |

---

## 4. Setup & Developer Runbook

### 4.1 Prerequisites
- Python `>= 3.10` (`scikit-learn`, `xgboost`, `torch`, `pandas`, `openpyxl`, `fastapi`, `uvicorn`, `pydantic`)
- Node.js `>= 18` & `npm`
- Optional: `GEMINI_API_KEY` environment variable in `backend/node-api/.env` for live `/api/chat` responses

### 4.2 Running Path A — Three-Tier Enterprise Application
Start each tier in a dedicated terminal from repository root:

```powershell
# Terminal 1: Unified FastAPI Inference Service (:8000)
cd ml-service
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Node.js + Express Gateway & Agentic/Gemini Server (:3000)
cd backend/node-api
npm install
# Optional: set GEMINI_API_KEY in .env
npm run dev

# Terminal 3: React + Vite Dashboard (:5173)
cd frontend
npm install
npm run dev
```

### 4.3 Running Path B — Reproducible Benchmark & Static Browser Lab (`submission/`)
```powershell
# Re-run full 32-run benchmark + browser weight export + verification
python submission/scripts/run_pipeline.py
python submission/scripts/make_figures.py
python submission/scripts/export_browser.py
python submission/scripts/verify.py

# Serve Claim Fraud Lab locally on :8080
python submission/scripts/serve.py
```

---

## 5. API & Schema Reference

### 5.1 FastAPI Model Inference (`ml-service` on `http://127.0.0.1:8000`)
- `GET /health`: Service readiness + loaded Classical & Deep Learning model inventory.
- `POST /predict/classical/{model_name}` where `model_name ∈ {logistic_regression, random_forest, xgboost, svm, gradient_boosting}` (or UI alias mapping via Express).
- `POST /predict/dl/{model_name}` where `model_name ∈ {mlp, residual_mlp, cnn1d, tabtransformer, autoencoder}`.

**Standard Request Body (`ClaimFeaturesRequest`):**
```json
{
  "ClaimAmount": 8500.0,
  "PatientAge": 45,
  "PatientIncome": 48000.0,
  "ClaimDate": "2024-03-15",
  "PatientGender": "M",
  "ProviderSpecialty": "Cardiology",
  "ClaimStatus": "Pending",
  "PatientMaritalStatus": "Married",
  "PatientEmploymentStatus": "Employed",
  "ClaimType": "Inpatient",
  "ClaimSubmissionMethod": "Online",
  "Cluster": 1
}
```

### 5.2 Node.js Express Gateway (`backend/node-api` on `http://127.0.0.1:3000`)
- `GET /health`: Gateway health & downstream connectivity.
- `POST /api/ml/predict`: Proxies payload cleanly to `:8000` classical or DL endpoint.
- `POST /agentic/predict`: Executes `AgenticEnsemble-v1` (4 trace events: `ingestion`, `validation`, `risk-scorer`, `xai-explainer`) over 9 intake attributes.
- `POST /api/chat`: Calls Google Gemini (`gemini-2.0-flash`) with domain-grounded claim context (`backend/node-api/src/routes/chat.ts`).

---

## 6. Documented Limitations & Stale Artifact Discrepancies (Honest Engineering Notes)

1. **`Cluster` Leakage in Track A vs. Track B:** Track A keeps `Cluster` (`29f`) strictly to match historical checkpoint shapes and API schemas; real-time claim intake before post-hoc clustering MUST use Track B's `Intake (28f)` pipeline or drop `Cluster`.
2. **`ClaimAmount` Single-Feature Sensitivity:** Dropping `ClaimAmount` in `no_amount` lowers every classifier from `~1.0000` AP down to `0.1869–0.2404` AP because dataset positive labels never occur below `$7,465.07`. Future synthetic/real clinical datasets must inject low-dollar anomalies (e.g., high-frequency upcoding) to avoid amount shortcuts.
3. **Uncalibrated Output Scores:** Both neural logits passed through `sigmoid` under `pos_weight=15.67` and tree/Platt outputs represent ranking propensities rather than well-calibrated posterior probabilities (`CalibratedClassifierCV` is recommended before monetary expected-loss multiplication).
4. **Approach 3 Runtime vs. Research UI Separation:** `backend/node-api/src/agentic/predict.ts` currently returns a fast deterministic heuristic formula rather than calling `:8000` FastAPI ML/DL models or running LangGraph/RAG/OCR pipelines; `Approach-3-Agentic-AI/app/services/xai_service.py` provides an offline XAI/SHA-256 scaffold awaiting full service unification.
5. **Submodule README Updates:** Legacy developer notes inside `Approch-2-DL/README.md` and `backend/node-api/README.md` reflect earlier sprint snapshots; readers should rely on this unified documentation suite (`docs/`), `PROJECT_PROGRESS.md`, `DECISIONS.md`, and live source code.
