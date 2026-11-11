# Volume 06 — Full-Stack Three-Tier Platform Architecture (`ml-service/`, `backend/node-api/`, `frontend/`)

> **Document Classification:** Production Integration Architecture, Tier Boundary Specification & UI State Audit  
> **Primary Ownership:** **M. Jagadeeswar (`23BDS033`)** (3-tier platform, ML/DL services, API client, React ML/DL/Overview pages) & **B. Varshith (`23BDS011`)** (`predict.ts`, `chat.ts`, Agentic UI pages & widgets).  
> **Live Service Topology:** Tier 1 FastAPI `:8000` (`ml-service/`) ← Tier 2 Express TypeScript `:3000` (`backend/node-api/`) ← Tier 3 React 19 + Vite `:5173` (`frontend/`).  
> **Primary Ground-Truth Sources:** [`README.md`](../../README.md), [`ml-service/app/main.py`](../../ml-service/app/main.py), [`claims.py`](../../ml-service/app/schemas/claims.py), [`predictions.py`](../../ml-service/app/api/predictions.py), [`classical_service.py`](../../ml-service/app/services/classical_service.py), [`dl_service.py`](../../ml-service/app/services/dl_service.py), [`mlp_service.py`](../../ml-service/app/services/mlp_service.py), [`smoke_test.py`](../../ml-service/smoke_test.py), [`backend/node-api/src/server.ts`](../../backend/node-api/src/server.ts), [`routes/ml.ts`](../../backend/node-api/src/routes/ml.ts), [`routes/chat.ts`](../../backend/node-api/src/routes/chat.ts), [`agentic/predict.ts`](../../backend/node-api/src/agentic/predict.ts), [`shared/schemas.ts`](../../backend/node-api/src/shared/schemas.ts), [`frontend/vite.config.ts`](../../frontend/vite.config.ts), [`frontend/src/App.tsx`](../../frontend/src/App.tsx), [`frontend/src/types.ts`](../../frontend/src/types.ts), [`frontend/src/lib/api.ts`](../../frontend/src/lib/api.ts), and [`frontend/src/pages/*.tsx`](../../frontend/src/pages/HomePage.tsx).

---

## 1. Platform Integration Rationale & Three-Tier Separation of Concerns

Serving heterogeneous scientific artifacts (scikit-learn `ColumnTransformer` pipelines, XGBoost boosters, PyTorch tensor models, heuristic agentic chains, and Gemini SDK streams) directly from a browser SPA requires isolating **runtime domains**, **dependency ecosystems**, and **fault zones** across three loosely coupled processes:

1. **Tier 1 (`ml-service/`, Python FastAPI `:8000`) — Pure Numerical Inference Engine:**
   Encapsulates heavy C++/Python scientific runtimes (`numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `joblib`). Exposes strongly validated REST endpoints strictly executing preprocessing transformations and loaded model forward passes without database overhead, CORS confusion, or generative AI credentials.
2. **Tier 2 (`backend/node-api/`, Node Express TypeScript `:3000`) — Web Gateway, Orchestration & AI Facade:**
   Provides a uniform JSON gateway to the browser (`CORS`, shared schemas, graceful upstream error normalization), proxies `/api/ml/*` cleanly down to Python `:8000`, runs lightweight in-process TypeScript routes (`/api/agentic/predict`), and isolates external LLM calls (`POST /api/chat` → `gemini-3.8-flash`) with `.env` secret management.
3. **Tier 3 (`frontend/`, React 19 + TypeScript + Tailwind + Vite `:5173`) — Presentation & Interactive Screening Workspace:**
   Delivers four top-level tabs (`Overview`, `Classical ML`, `Deep Learning`, `Agentic AI`), interactive preset claim loaders, multi-model batch comparison dashboards, offline graceful fallback logic when backends are stopped, and the conditional `ChatBox` analyst drawer.

```text
+---------------------------------------------------------------------------------------------------+
|                        THREE-TIER RUNTIME PROCESS TOPOLOGY (TRACK A)                              |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  TIER 3: Browser SPA (React 19 + Vite, Port :5173)                                                |
|  |-- src/App.tsx (63L): Active Tab Router -> HomePage | ClassicalMLPage | DLPage | AgenticPage    |
|  |-- Conditionally mounts ChatBox.tsx (309L) strictly when activeTab === "agentic"                |
|  `-- src/lib/api.ts (172L) + vite.config.ts (19L) -> HTTP requests to :3000 /api/*                |
|                                         |                                                         |
|                                         v  HTTP/JSON (Vite Dev Proxy / Direct CORS)               |
|  TIER 2: Express TypeScript API Gateway (Node, Port :3000)                                        |
|  |-- src/server.ts (21L): App bootstrap, JSON middleware, route mounting                          |
|  |-- src/routes/health.ts:  GET  /api/health                                                      |
|  |-- src/routes/ml.ts (111L): POST /api/ml/predict[/ml|/dl] --------+                             |
|  |-- src/agentic/predict.ts (39L): POST /api/agentic/predict        | (Handled in-process Node)   |
|  `-- src/routes/chat.ts (47L): POST /api/chat ----------------------+ (Calls Google Gemini 3.8)   |
|                                                                     |                             |
|                                                                     v HTTP Proxy -> 127.0.0.1:8000|
|  TIER 1: Python FastAPI ML/DL Microservice (Uvicorn, Port :8000)                                  |
|  |-- app/main.py (8L) + app/api/predictions.py (62L)                                              |
|  |-- app/schemas/claims.py (56L): Strict Pydantic v2 ClaimRequest (extra="forbid")                |
|  |-- app/services/classical_service.py (117L) -> loads Approch-1/models/*.joblib + feature_names  |
|  `-- app/services/dl_service.py (187L) + mlp_service.py (146L) -> loads Approch-2-DL/models/*.pt  |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Tier 1 Specification: Python FastAPI Microservice (`ml-service/`, Port `:8000`)

### 2.1 File Structure & Responsibilities (`ml-service/`)

| File | Lines | Exact Architectural Role |
| :--- | :---: | :--- |
| `ml-service/README.md` | `36` | Installation & uvicorn launch commands |
| `ml-service/requirements.txt` | `7` | Slim serving dependency list (FastAPI, Uvicorn, Pydantic, PyTorch, XGBoost, scikit-learn) |
| `ml-service/app/main.py` | `8` | Creates FastAPI instance and registers prediction router cleanly |
| `ml-service/app/schemas/claims.py` | `56` | Defines strict input payload models (`ClaimRequest` with `ConfigDict(extra="forbid")`) & structured response schemas |
| `ml-service/app/api/predictions.py` | `62` | Exposes `POST /predict/ml`, `POST /predict/dl`, and `POST /predict` |
| `ml-service/app/services/classical_service.py` | `117` | Loads `preprocessor.joblib`, `feature_names.txt`, and all `5` classical `.joblib` estimators; runs aligned inference |
| `ml-service/app/services/dl_service.py` | `187` | Loads all `5` PyTorch `*_best.pt` weights + `*_metrics.json` thresholds; computes probabilities & Autoencoder sigmoid scaling |
| `ml-service/app/services/mlp_service.py` | `146` | Modular/focused MLP service loader & preprocessor alignment helper |
| `ml-service/smoke_test.py` | `46` | Programmatic FastAPI `TestClient` / HTTP verifier testing live route behavior & model outputs |

### 2.2 Strict Input Schema Enforcement (`claims.py`, `56` lines)

[`ml-service/app/schemas/claims.py`](../../ml-service/app/schemas/claims.py) prevents silent schema drift — one of the most dangerous production failure modes in scikit-learn/PyTorch pipelines where unexpected JSON keys (`claimAmount` instead of `ClaimAmount`, or extra client debug fields) get silently ignored while defaults take over:

* **`extra="forbid"` Guard:** By configuring Pydantic with `extra="forbid"`, any inbound request containing misspelled, camelCase, or undeclared attributes is immediately rejected at FastAPI parameter parsing with **HTTP `422 Unprocessable Entity`** before touching numpy arrays or neural layers.
* **Required Raw Domain Properties Aligned to Track A Preprocessor:**
  * Numeric & Calendar Fields: `ClaimAmount`, `PatientAge`, `PatientIncome`, `ClaimDate` (`YYYY-MM-DD` string parsed into `ClaimYear`, `ClaimMonth`, `ClaimDayOfWeek`), `Cluster` (`0..3`).
  * Categorical Fields: `PatientGender`, `ProviderSpecialty`, `PatientMaritalStatus`, `PatientEmploymentStatus`, `ClaimType`, `ClaimSubmissionMethod`.

### 2.3 Endpoint Matrix (`app/api/predictions.py`, `62` lines)

| Route Path | HTTP Method | Target Service Layer | Behavior & Model Routing |
| :--- | :---: | :--- | :--- |
| **`POST /predict/ml`** | `POST` | `classical_service.py` | Scores payload using specified Approach 1 `.joblib` estimator (`LogisticRegression`, `RandomForest`, `XGBoost`, `SVM`, `GradientBoosting`) or compares across models. |
| **`POST /predict/dl`** | `POST` | `dl_service.py` | Transforms payload via Approach 1 `preprocessor.joblib`, casts to `torch.float32` `(1, 29)`, and runs specified Approach 2 `.pt` model (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`, `Autoencoder`). |
| **`POST /predict`** | `POST` | Default / Unified Handler | Primary top-level prediction route unifying ML inference cleanly behind a single REST endpoint. |

### 2.4 Classical Inference Alignment (`classical_service.py`, `117` lines)

1. Loads `Approch-1/models/preprocessor.joblib` and reads exact `29` ordered labels from `Approch-1/data/processed/feature_names.txt`.
2. Decomposes `ClaimDate` via `pd.to_datetime(...).dt.{year, month, dayofweek}` matching `Approch-1/scripts/preprocess.py` (`0 = Monday` .. `6 = Sunday`).
3. Builds a column-ordered DataFrame, transforms via `preprocessor.transform(df)`, and calls `.predict_proba(...)[:, 1]` for calibrated probability estimation. Note that `SVM` in `Approch-1/models/SVM.joblib` uses Platt-scaled probabilities (`39` test positives behavior) consistently under `.predict_proba()`, eliminating the `.predict()` (`51` flagged) divergence inside live serving.

### 2.5 Deep Learning Inference & Autoencoder Scaling (`dl_service.py`, `187` lines & `mlp_service.py`, `146` lines)

All neural inference runs under `model.eval()` and `torch.no_grad()` context managers so `Dropout(0.2)` / `Dropout(0.1)` layers disable stochastic zeroing deterministically:

* **Supervised PyTorch Classifiers (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`):** Pass raw logit $z$ through standard logistic sigmoid $p = 1 / (1 + e^{-z})$ and evaluate binary decision against `0.5` cutoff. Remember under `ADR-05` that because training used `BCEWithLogitsLoss(pos_weight = 2961/189)`, raw $p$ reflects shifted log-odds ($+\ln 15.67$) rather than true empirical prevalence calibration.
* **Unsupervised `Autoencoder` Inference Scaling (`dl_service.py` verbatim formula):**
  Computes per-claim mean squared error $e = \frac{1}{29}\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$, loads exact `threshold` ($\tau_{0.95} = 0.12858837842941284$) from `Approch-2-DL/results/autoencoder_metrics.json`, decides `is_fraud = (recon_error >= threshold)`, and calculates smooth bounded risk score:
  $$\text{score} = \frac{1}{1 + \exp\!\left(\frac{-(\text{recon\_error} - \text{threshold}) \cdot 5}{\max(\text{threshold}, 10^{-8})}\right)}$$
  Notice why `max(threshold, 1e-8)` guards against zero division and factor `5` ensures zero-error claims (`recon_error = 0`) land cleanly near `0.0067` (`sigma(-5)`) while `recon_error == threshold` maps identically to `0.5000`.

### 2.6 Microservice Verification (`smoke_test.py`, `46` lines)

`ml-service/smoke_test.py` exercises route serialization, `ClaimRequest` validation, and model loading end-to-end across valid claims without requiring manual curl construction.

---

## 3. Tier 2 Specification: Node.js + Express TypeScript Gateway (`backend/node-api/`, Port `:3000`)

### 3.1 File Inventory & Server Initialization (`src/server.ts`, `21` lines)

| File | Lines | Architectural Role |
| :--- | :---: | :--- |
| `backend/node-api/package.json` | `24` | TypeScript, Express, CORS, `dotenv`, `@google/generative-ai`, `tsx` runtime dependencies |
| `backend/node-api/README.md` | `33` | Setup, `.env` expectations, script commands |
| `backend/node-api/src/server.ts` | `21` | Express initialization, middleware configuration (`express.json`, CORS), mounts `/api/*` routers, binds `:3000` |
| `backend/node-api/src/shared/schemas.ts` | `25` | Shared TypeScript interfaces / request validation helpers across Express routes |
| `backend/node-api/src/routes/health.ts` | — | Readiness probe endpoint (`GET /api/health`) |
| `backend/node-api/src/routes/ml.ts` | `111` | HTTP gateway proxy routing `/api/ml/*` calls cleanly to Tier 1 `http://127.0.0.1:8000` |
| `backend/node-api/src/agentic/predict.ts` | `39` | Local Express agentic evaluation handler (`POST /api/agentic/predict`, `AgenticEnsemble-v1`) |
| `backend/node-api/src/routes/chat.ts` | `47` | Gemini LLM conversational handler (`POST /api/chat`, `gemini-3.8-flash`) |

### 3.2 Request Proxy Plane (`src/routes/ml.ts`, `111` lines)

Rather than duplicating scikit-learn transformations inside JavaScript, `ml.ts` acts as an API translation & fault-isolation proxy:

* Accepts structured JSON requests from React at `/api/ml/predict`, `/api/ml/predict/ml`, and `/api/ml/predict/dl` (supporting single-estimator inference and comparison requests).
* Forwards payloads over loopback HTTP to Tier 1 (`127.0.0.1:8000`).
* Propagates HTTP `422` validation diagnostics from FastAPI cleanly back to browser UI components while converting network connection failures (`ECONNREFUSED` when `:8000` is down) into structured gateway error JSON so the frontend can surface degraded-mode warnings or fallback behavior reliably.

### 3.3 In-Process Agentic Shadow Route (`src/agentic/predict.ts`, `39` lines)

As audited in Volume 05, requests sent by React `AgenticPage.tsx` to `POST /api/agentic/predict` do **not** proxy to `:8020` (`Approach-3-Agentic-AI`) or `:8000` (`ml-service`): instead `predict.ts` evaluates `min(max((claim_amount * 0.01) + (patient_age * 0.005) - 2.0, 0.0), 1.0)` directly in TypeScript and returns `model_name: "AgenticEnsemble-v1"`, `agent_id: "agent-chain-001"`, `feature_count: 9`, static feature attributions, and the 4-stage `audit_log` sequence.

### 3.4 Conversational Copilot Plane (`src/routes/chat.ts`, `47` lines)

1. Resolves `.env` explicitly from **`../../Approach-3-Agentic-AI/.env`** using `dotenv`.
2. Reads `GEMINI_API_KEY` and constructs `GoogleGenerativeAI`.
3. Targets model identifier **`"gemini-3.8-flash"`** (`line 13` of `chat.ts`) to answer analyst queries grounded in claim context serialized by `ChatBox.tsx`.

---

## 4. Tier 3 Specification: React 19 + TypeScript + Tailwind SPA (`frontend/`, Port `:5173`)

### 4.1 Frontend Architecture, Build Tooling & Live Module Map

| File | Lines | Mounting Status (`App.tsx`) | Architectural Role & Verified Behavior |
| :--- | :---: | :--- | :--- |
| `frontend/package.json` | `22` | Build manifest | React `19`, TypeScript, Vite, Tailwind CSS dependencies |
| `frontend/index.html` | `15` | Entry HTML | Root `<div id="root"></div>` container & metadata |
| `frontend/vite.config.ts` | `19` | Vite dev server config | Configures React plugin & local `/api` dev proxy toward `:3000` |
| `frontend/src/App.tsx` | `63` | **Root Router** | Manages `activeTab` across `HomePage`, `ClassicalMLPage`, `DLPage`, `AgenticPage`; renders `ChatBox` only on `agentic` |
| `frontend/src/types.ts` | `82` | Shared TS contracts | Strong TypeScript payload & response types shared across pages and `api.ts` |
| `frontend/src/lib/api.ts` | `172` | API client module | Typed `fetch` wrappers calling Express `:3000` routes (`health`, ML, DL, Agentic, Chat) |
| `frontend/src/pages/HomePage.tsx` | `167` | **Mounted** (`overview` tab) | Project landing overview, approach navigation cards, architectural highlights |
| `frontend/src/pages/ClassicalMLPage.tsx` | `405` | **Mounted** (`ml` tab) | Interactive claim scoring + 5-model classical comparison & offline demo fallback |
| `frontend/src/pages/DLPage.tsx` | `349` | **Mounted** (`dl` tab) | Neural architecture inspector + live scoring across `MLP`, `ResMLP`, `CNN1D`, `TabTransformer`, `Autoencoder` |
| `frontend/src/pages/AgenticPage.tsx` | `247` | **Mounted** (`agentic` tab) | Active agentic pipeline trigger UI + trace viewer + XAI bar visualization |
| `frontend/src/agentic/components/ChatBox.tsx` | `309` | **Mounted** (`agentic` tab) | Conversational copilot panel wired to `POST /api/chat` |
| `frontend/src/agentic/data/agents.ts` | `56` | Data catalog | Stores `6` conceptual specialist role definitions |
| `frontend/src/agentic/pages/AgenticDashboard.tsx` | `142` | **Unmounted** | Standalone alternative dashboard module present on disk only |
| `frontend/src/agentic/pages/ClaimInputForm.tsx` | `260` | **Unmounted** | Standalone form widget under `src/agentic/pages/` |
| `frontend/src/agentic/components/{AuditTrail.tsx, AgentCard.tsx, ScoreGauge.tsx}` | `157+` | **Unmounted** | Modular card/timeline widgets present on disk (`App.tsx` renders `AgenticPage.tsx` instead) |
| `frontend/src/components/{PipelinePreview.tsx, ThemeToggle.tsx}` | — | **Unmounted** | Standalone UI components on disk not imported in `App.tsx` |
| `frontend/src/lib/{useApiHealth.ts, useScrollReveal.ts, useTheme.ts}` | — | **Unmounted hooks** | Utility React hooks on disk unused by root `App.tsx` |

### 4.2 Detailed Tab Workflows & Offline Fallback Behavior (`ClassicalMLPage.tsx`)

Inspecting [`frontend/src/pages/ClassicalMLPage.tsx`](../../frontend/src/pages/ClassicalMLPage.tsx) (`405` lines) reveals an explicit UI design priority (`ADR-13`): if `:3000` or `:8000` is offline during classroom presentation or static frontend inspection, the page catches API fetch failures cleanly and executes a client-side fallback heuristic on lines `92–98` instead of crashing:

$$p_{\text{fallback}}(\text{ClaimAmount}, \text{PatientAge}) = \operatorname{clip}\Big((\text{ClaimAmount} \times 0.00003) + (\text{PatientAge} \times 0.001) - 1.5,\; 0.01,\; 0.99\Big)$$

#### Proof of `0.01` Clamp Saturation on Realistic Claims

Notice the inverse failure pathology relative to Approach 3's Node heuristic:

* Lower clamp `0.01` triggers whenever:
  $$(\text{ClaimAmount} \times 0.00003) + (\text{PatientAge} \times 0.001) \le 1.51$$
* For maximum adult age `PatientAge = 99`:
  $$0.00003 \cdot \text{ClaimAmount} \le 1.51 - 0.099 = 1.411 \implies \text{ClaimAmount} \le \mathbf{\$47,033.33}$$
* At `PatientAge = 40`: `ClaimAmount <= \$49,000.00`; at `PatientAge = 0`: `ClaimAmount <= \$50,333.33`.
* Because **every claim in `Health Insurance Fraud Claims.xlsx` falls strictly between `\$100.12` and `\$9,997.20`** (`4.7×` below the `\$47,033.33` activation threshold!), **offline mode in `ClassicalMLPage.tsx` evaluates every workbook record identically to floor value `0.01`** unless a user manually injects extreme `>\$47k` synthetic test inputs or starts the live `:8000`/`:3000` servers!

This mathematical discovery highlights why running the full 3-terminal stack (`:8000` + `:3000` + `:5173`) is mandatory for authentic ML scoring validation.

### 4.3 Live Integration End-to-End Verification Matrix (`PROJECT_PROGRESS.md` & `README.md`)

| Verification Check | Verified Target | Expected Outcome When All 3 Tiers Are Online |
| :--- | :--- | :--- |
| **Root Health Check** | `GET http://127.0.0.1:8000/health` & `GET http://127.0.0.1:3000/api/health` | Both Python FastAPI and Express Node report active readiness |
| **Live Spot-Check (`Index 2`, Legit)** | Workbook row `Index 2` passed through `:3000` → `:8000` | Classified Legitimate across Classical ML & supervised DL models |
| **Live Spot-Check (`Index 0`, Fraud)** | Workbook row `Index 0` passed through `:3000` → `:8000` | Classified High-Risk Fraud matching ground truth label |
| **Strict Schema Negative Guard** | Payload with unknown extra JSON field sent to `POST :8000/predict/ml` | Rejected with `HTTP 422 Unprocessable Entity` (`extra="forbid"`) |
| **Agentic & Copilot Flow** | `POST :3000/api/agentic/predict` + `POST :3000/api/chat` | Returns 4-stage trace + `gemini-3.8-flash` explanation (requires valid key in `Approach-3-Agentic-AI/.env`) |

---

## 5. Multi-Terminal Startup & Live Smoke Verification Runbook (`README.md`)

Launching all three production tiers simultaneously on Windows requires three isolated PowerShell terminals:

```powershell
# Terminal 1 — Tier 1 Python ML & DL Microservice (:8000)
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\ml-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Tier 2 Node.js Express API Gateway (:3000)
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\backend\node-api
npm install
npm run dev

# Terminal 3 — Tier 3 React + Vite Frontend (:5173)
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\frontend
npm install
npm run dev
```

### 5.1 Automated Endpoint Smoke Check (`ml-service/smoke_test.py` + PowerShell Check)

```powershell
# Verify Python ML microservice internals offline/via TestClient or local check
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\ml-service
.\.venv\Scripts\python.exe smoke_test.py

# Verify Tier 2 -> Tier 1 live health cleanly
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
Invoke-RestMethod -Uri "http://127.0.0.1:3000/api/health" -Method Get
```

---

## 6. Three-Tier Security, Resilience & Production Hardening Analysis

| Layer / Concern | Current Development State | Production Hardening Recommendation |
| :--- | :--- | :--- |
| **Request Input Validation** | `ml-service` enforces `extra="forbid"`; `node-api` forwards JSON; `frontend` handles PascalCase types. | Mirror identical Zod runtime validation on `:3000` `ml.ts`, `predict.ts`, and `chat.ts` before forwarding upstream or calling Gemini. |
| **Secret Isolation (`GEMINI_API_KEY`)** | `chat.ts` reaches across directories into `../../Approach-3-Agentic-AI/.env`. | Relocate runtime `.env` cleanly to `backend/node-api/.env` or container environment variables to decouple from Python prototype directories. |
| **Agentic Scoring Discrepancy** | `/api/agentic/predict` runs heuristic stub instead of calling `:8000`. | Route Tier 2 `predict.ts` risk-scorer stage to `:8000/predict/ml` (`XGBoost` or Track B `RandomForest` `intake`), preserving audit trace UX with real model accuracy. |
| **Client-Side Demo Fallbacks** | `ClassicalMLPage.tsx` clamps workbook claims to `0.01`; `AgenticPage.tsx` echoes heuristic `1.0`. | Surface unmistakable "OFFLINE DEMO MODE" banner and align offline preview coefficients to realistic domain priors (`\$7,465+` regime). |
| **Observability & CORS** | Local dev URLs (`127.0.0.1:8000`, `:3000`, `:5173`); unstructured console logging. | Add structured request-ID correlation (`X-Request-ID`) spanning `:5173 → :3000 → :8000` + response-latency histograms. |
