# Mini-Project 2 — Project Progress

## Final two-approach submission 20 October 2026

The completed historical submission package is documented in `submission/README.md`.

- Approach 1: all five classical algorithms retrained and evaluated.
- Approach 2: MLP, Residual MLP and 1D CNN trained and evaluated.
- Four experiments compare the original contract, removal of Cluster, removal of Cluster plus amount, and chronological intake validation.
- The separate model laboratory supports both approaches with real browser inference and an independent local Python API (`submission/scripts/serve.py` on port 8080).
- Final research manuscript, technical report and presentation are in `submission/deliverables/`.
- Raw data and historical model artifacts remain intact. The earlier application skeleton is preserved.

> **Note:** The historical `submission/` package remains distinct from the full-stack web application documented below.

---

## Reference Materials

- `Health Insurance Fraud Claims.xlsx`
- `Medical_Insurance_Claim_Fraud_Detection_Project_Statement.pdf`
- `Mini_Project_2_Full_Low_Level_Technical_Specification.docx`

These are project reference/source documents. They are not to be modified as part of implementation.

---

## Team Responsibilities

- **Approach 1 (Classical ML):** Ganesh (Model training, baseline evaluation). Integration via shared APIs maintained by Jagadeeswar.
- **Approach 2 (Deep Learning):** Jagadeeswar (Architectures, training loops, checkpoints, metrics, and inference service).
- **Approach 3 (Agentic AI):** Varshith (Agent orchestration, reasoning graph, and internal agent implementation). Shared API routes and dashboard UI wrapper maintained for common integration without modifying Varshith's internal files.
- **Common Website & API Integration:** Jagadeeswar (React frontend, Node.js proxy API, FastAPI inference services).

---

## Current Architecture & Website Integration

The integrated platform operates as a 3-tier architecture:

```text
React (Vite + TypeScript @ Port 5173)
        ↓  (Proxy /api and /agentic)
Node.js + Express (@ Port 3000)
        ↓  (Upstream HTTP)
Python + FastAPI (`ml-service` @ Port 8000)
        ↓
Classical ML (joblib)  |  Deep Learning (PyTorch)  |  Agentic AI (/agentic/predict)
```

### Verified Endpoints & Services

1. **Frontend (`frontend/`)**:
   - Modern tabbed single-page application built with React and TypeScript (`npm run build` passes with zero errors).
   - **Dashboard (`HomePage.tsx`)**: High-level platform KPIs, approach summaries, and dynamic stats loaded from results APIs.
   - **Classical ML Page (`ClassicalMLPage.tsx`)**: Model selector (`LogisticRegression`, `RandomForest`, `XGBoost`, `SVM`, `GradientBoosting`), live prediction form submitting to `/api/ml/predict`, and a benchmark table rendering verified metrics from `model_results.json`.
   - **Deep Learning Page (`DLPage.tsx`)**: Architecture selector (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`, `Autoencoder`), dynamic test metrics banner, live inference form submitting to `/api/dl/predict?model=...`, and distinct Autoencoder explanation notes.
   - **Agentic AI Page (`AgenticPage.tsx`)**: Claim risk assessment triggering multi-agent pipeline and visualizing the chain-of-custody audit trail.

2. **Backend Gateway (`backend/node-api/`)**:
   - Express server on port 3000 (`tsc` passes with zero errors).
   - `POST /api/ml/predict?model=...`: Forwards claim payloads to FastAPI `/predict/ml`.
   - `POST /api/dl/predict?model=...`: Forwards claim payloads to FastAPI `/predict/dl`.
   - `GET /api/results/ml`: Serves verified metrics from `Approch-1/evaluation/model_results.json`.
   - `GET /api/results/dl`: Serves baseline DL metrics (`mlp_metrics.json`).
   - `GET /api/results/dl/all`: Serves consolidated metrics for all 5 DL models.
   - `POST /agentic/predict`: Serves the multi-agent ensemble audit trail pipeline.

3. **FastAPI ML Service (`ml-service/`)**:
   - Uvicorn service on port 8000 using Python 3.13 in `Approch-2-DL/.venv`.
   - `classical_service.py`: Caches Ganesh's preprocessor (`preprocessor.joblib`) and loads all 5 scikit-learn/XGBoost models from `Approch-1/models/`.
   - `dl_service.py`: Loads and caches all 5 PyTorch model checkpoints and evaluates them against the frozen 29-feature contract.

---

## Verified Model Benchmarks

All benchmark metrics below represent the exact values stored in the evaluation result artifacts.

### Approach 1 — Classical ML (`Approch-1/evaluation/model_results.json`)

Evaluated on the held-out test split (675 samples, 40 frauds, 635 legitimate):

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|
| **RandomForest** | 99.85% | 100.00% | 97.50% | **98.73%** | 99.996% | **99.94%** |
| **XGBoost** | 99.85% | 97.56% | 100.00% | **98.77%** | 100.00% | **100.00%** |
| **GradientBoosting** | 99.85% | 100.00% | 97.50% | **98.73%** | 98.47% | **97.65%** |
| **SVM** | 97.78% | 74.51% | 95.00% | 83.52% | 99.55% | 92.52% |
| **LogisticRegression** | 96.44% | 62.50% | 100.00% | 76.92% | 99.50% | 91.86% |

### Approach 2 — Deep Learning (`Approch-2-DL/results/*_metrics.json`)

All models evaluated on the frozen 29-feature contract (`X_test.npy`, `y_test.npy`):

| Model | Architecture | Best Epoch | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---|---|---|
| **MLP** | Feedforward Baseline | 34 | 98.81% | 84.78% | 97.50% | **90.70%** | 99.69% | 95.38% |
| **ResidualMLP** | Deep Residual | 20 | 98.52% | 80.00% | 100.00% | **88.89%** | 99.78% | 95.61% |
| **CNN1D** | 1D Convolutional | 41 | 98.81% | 83.33% | 100.00% | **90.91%** | 99.85% | **98.82%** |
| **TabTransformer** | Self-Attention | 34 | 98.67% | 81.63% | 100.00% | **89.89%** | 99.87% | **99.11%** |
| **Autoencoder** | Reconstruction Anomaly | 80 | 90.37% | 3.70% | 2.50% | **2.99%** | 46.02% | **5.23%** |

---

## Autoencoder Anomaly Detection Interpretation

The Autoencoder (DL-05) differs fundamentally from the other four supervised deep learning architectures:

- **Unsupervised Objective:** Trained solely on legitimate (non-fraud) claims to reconstruct normal feature patterns using MSE loss.
- **Decision Threshold:** A reconstruction error threshold of **0.1286** was fitted at the 95th percentile of normal validation samples.
- **Output Score Interpretation:** The score returned by the API is a normalized reconstruction error, **not a calibrated probability of fraud**.
- **Fraud Detection Performance:** With a test F1 of **2.99%** and test PR-AUC of **5.23%**, the Autoencoder exhibits poor fraud-detection performance compared to supervised baselines. This confirms that general feature reconstruction discrepancies do not correlate reliably with targeted insurance fraud indicators in this feature contract.

---

## Ground-Truth Prediction Validation

Two specific held-out test examples from `Approch-1/data/processed/X_test.csv` were tested end-to-end through the live proxy chain (`Node.js → FastAPI`):

1. **Known Legitimate Sample (`Index 2`, Ground Truth Label = 0)**:
   - `RandomForest`: Legitimate (`0.0833`)
   - `XGBoost`: Legitimate (`0.0006`)
   - `MLP`: Legitimate (`0.0000`)
   - `ResidualMLP`: Legitimate (`0.0000`)
   - `CNN1D`: Legitimate (`0.0000`)
   - `TabTransformer`: Legitimate (`0.0001`)
   - `Autoencoder`: Legitimate (`0.3694` anomaly score; reconstruction error did not exceed threshold)

2. **Known Fraudulent Sample (`Index 0`, Ground Truth Label = 1)**:
   - `RandomForest`: Fraud (`0.8133`)
   - `XGBoost`: Fraud (`0.9998`)
   - `MLP`: Fraud (`0.9615`)
   - `ResidualMLP`: Fraud (`0.9889`)
   - `CNN1D`: Fraud (`0.9894`)
   - `TabTransformer`: Fraud (`0.9893`)
   - `Autoencoder`: Legitimate (`0.4107` anomaly score; rose from 0.3694 but remained below threshold)

> **Validation Note:** While both supervised approaches correctly separated these two spot-check examples, two isolated predictions do not prove broad operational reliability. Furthermore, the Autoencoder failed to flag the fraudulent sample, highlighting its limitations as an unassisted fraud detector.

---

## Agentic AI Status (Approach 3)

- **Ownership:** Varshith owns the internal Agentic AI reasoning engine and agent graph in `Approach-3-Agentic-AI/`.
- **Integration Status:** The common application provides integration points via `POST /agentic/predict` and the frontend `AgenticPage.tsx` interface.
- **Verification:** Verified that `/agentic/predict` returns an audit trail with four pipeline stages (`ingestion`, `validation`, `risk-scorer`, `xai-explainer`). Varshith's internal implementations remain unmodified.

---

## Current Overall Status

### Completed Implementation
- [x] Reference data and problem statement specifications preserved.
- [x] Approach 1 Classical ML: 5 models trained, saved, and served via FastAPI (`RandomForest`, `GradientBoosting`, `LogisticRegression`, `SVM`, `XGBoost`).
- [x] `xgboost` package installed in active Python 3.13 venv; live inference confirmed.
- [x] Approach 2 Deep Learning: 5 architectures implemented, trained, and saved with checkpoints and metrics (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`, `Autoencoder`).
- [x] Multi-model DL inference service implemented in `ml-service/app/services/dl_service.py`.
- [x] FastAPI predictions router supporting `/predict/dl`, `/predict/ml`, and `/predict`.
- [x] Node.js backend proxy with routes for DL, ML, results, and agentic predictions.
- [x] React + TypeScript frontend with dynamic model selectors and real-time inference.
- [x] Zero compilation/build errors across both frontend (`vite build`) and backend (`tsc`).

### Demonstration Readiness vs. Model Implementation
- **Model Implementation & APIs:** Complete. All 10 ML/DL models and results endpoints are functional and verified.
- **Demonstration Readiness:** Final rehearsal of multi-service startup scripts and cross-team presentation preparation remaining before professor demonstration.

---

## Git State

Based on `git status --short`, changes in the working tree are **uncommitted** and the working tree is **not clean**:

- **Modified tracked files:**
  - `backend/node-api/package.json`, `package-lock.json`, `server.ts`, `src/agentic/predict.ts`
  - `frontend/index.html`, `vite.config.ts`, `src/App.tsx`, `src/styles.css`, `src/agentic/pages/*`
  - `ml-service/app/api/predictions.py`
  - `PROJECT_PROGRESS.md`
- **Untracked additions:**
  - `Approch-2-DL/models/{autoencoder, cnn1d, residualmlp, tabtransformer}/`
  - `Approch-2-DL/results/{autoencoder, cnn1d, residualmlp, tabtransformer}_metrics.json`
  - `Approch-2-DL/scripts/{dl_advanced_models.py, train_advanced.py, train_supervised.py}`
  - `backend/node-api/src/routes/ml.ts`, `src/shared/`
  - `frontend/src/pages/`, `src/types.ts`
  - `ml-service/app/services/{classical_service.py, dl_service.py}`
