# System Architecture: Frontend UI/UX, Backend Gateway & Inference Services

**Project:** Health Insurance Fraud Intelligence Platform (`Medical Insurance Claim Fraud Detection`)  
**Scope:** End-to-end developer & reviewer reference covering `frontend/`, `backend/node-api/`, `ml-service/`, and `Approach-3-Agentic-AI/`.

---

## 1. End-to-End Runtime Topology

```text
+-------------------------------------------------------------------------+
| React 19 + TypeScript + Vite SPA (http://127.0.0.1:5173)                |
|  - Dual Light/Dark Clinical Design System (styles.css + useTheme.ts)    |
|  - Pages: Overview, Classical ML (5), Deep Learning (5), Agentic AI     |
|  - Global Components: ClaimFieldsForm, PredictionResultCard, ChatBox    |
|  - Vite Proxy: /api -> :3000  and  /agentic -> :3000                    |
+------------------------------------+------------------------------------+
                                     | HTTP JSON
                                     v
+-------------------------------------------------------------------------+
| Node.js + Express + TypeScript Gateway (http://127.0.0.1:3000)          |
|  - GET  /api/health        -> { status: "ok" }                          |
|  - POST /api/ml/predict    -> Proxies ?model=... to FastAPI /predict/ml |
|  - POST /api/dl/predict    -> Proxies ?model=... to FastAPI /predict/dl |
|  - GET  /api/results/ml    -> Reads Approch-1/evaluation/model_results  |
|  - GET  /api/results/dl*   -> Reads Approch-2-DL/results/*_metrics.json |
|  - POST /agentic/predict   -> Runs AgenticEnsemble-v1 (4 audit stages)  |
|  - POST /api/chat          -> Google Gemini (@google/generative-ai)     |
+--------------------+----------------------------------------------------+
                     | HTTP JSON (ML/DL routes)
                     v
+-------------------------------------------------------------------------+
| Python FastAPI ML Service (http://127.0.0.1:8000, ml-service/)          |
|  - GET  /health                                                         |
|  - POST /predict/ml?model=<LR|RF|XGB|SVM|GradientBoosting>              |
|  - POST /predict/dl?model=<MLP|ResidualMLP|CNN1D|TabTransformer|AE>     |
|  - Loads Approch-1/data/processed/preprocessor.joblib (29 features)     |
+-------------------------------------------------------------------------+

Standalone Microservice Scaffold (Optional Port 8020):
- Approach-3-Agentic-AI/: XAIExplainer + SHA-256 tamper-evident audit_log
```

---

## 2. Frontend Architecture & UI/UX Engineering (`frontend/src/`)

### 2.1 Application Shell & Navigation (`App.tsx`)
* **Top Sticky Header (`.app-header`):**
  * **Brand Navigation (`logo` button):** Accessible button returning to the `Overview` workspace and scrolling cleanly without hash reload jumps.
  * **Main Navigation Tabs (`nav-tabs`):** Four top-level views (`Overview`, `Classical ML`, `Deep Learning`, `Agentic AI`) with `aria-current="page"` state indicators and responsive flex wrapping on smaller screens (`<= 760px`).
  * **Live Gateway Status Badge (`useApiHealth.ts`):** Polls `GET /api/health` every 30s (`checking`, `online`, `offline`) and displays `API Online` vs `Offline Demo` cleanly.
  * **Global AI Assistant Drawer Trigger + FAB (`ChatBox.tsx`):** Accessible from both the header (`AI Assistant` button) and persistent floating bottom-right button (`.chat-fab`), supporting Escape-to-close, focus restoration, auto-resizing multi-line composer, retry on transient failure, and prompt quick-chips.
  * **Theme Switcher (`ThemeToggle.tsx` + `useTheme.ts`):** Persists user preference (`light` clinical default or `dark` slate/teal `[data-theme="dark"]`) in `localStorage` (`hfi-theme`) with early inline hydration in `frontend/index.html` to eliminate flash-of-unstyled theme on reload.

### 2.2 Dual-Theme Clinical Design Tokens (`frontend/src/styles.css`)
All legacy inline hardcoded dark colors and conflicting cascade overrides were consolidated into a unified semantic CSS token table:
* **Light Mode (`:root`):** Clinical teal primary `#0f766e`, clean neutral background `#f3f7f8`, crisp `#ffffff` surfaces, deep slate `#12292f` primary ink ensuring `>= 7:1` text contrast.
* **Dark Mode (`[data-theme="dark"]`):** Deep slate `#0b1116` background, `#121b22` elevated panels, `#edf4f6` primary text, vibrant `#14b8a6` interactive accents with dark text (`#06211f`) on active pills for consistent WCAG AA/AAA readability.
* **Accessibility & Responsive Grid:**
  * Keyboard `:focus-visible` ring (`2px solid var(--border-hi)`).
  * `@media (prefers-reduced-motion: reduce)` honored cleanly.
  * Responsive breakpoints at `1040px`, `760px`, and `520px` stack KPI tiles (`5 -> 3 -> 2` columns), collapse 2-column claim forms (`2 -> 1`), convert the chat drawer into a mobile bottom sheet, and scroll wide benchmark tables smoothly inside `.table-wrap` without causing viewport horizontal overflow.

### 2.3 Page & Shared Component Breakdown

| File | Responsibility & UX Features |
|---|---|
| `frontend/src/data/benchmarks.ts` | Holds `STATIC_CLASSICAL_BENCHMARKS` (exact values from `Approch-1/evaluation/model_results.json`), `STATIC_DL_BENCHMARKS` (exact values from all 5 `Approch-2-DL/results/*_metrics.json` files), model metadata, 3 `CLAIM_PRESETS` (`Routine Legitimate $794`, `High-Risk Flagged $8,950`, `Boundary Case $7,620`), and deterministic `computeOfflineDemoPrediction()`. |
| `frontend/src/components/ClaimFieldsForm.tsx` | Shared 11-field intake form (`ClaimAmount`, `ClaimDate`, `PatientAge`, `PatientIncome`, `PatientGender`, `ProviderSpecialty`, `PatientMaritalStatus`, `PatientEmploymentStatus`, `ClaimType`, `ClaimSubmissionMethod`, `Cluster`) with preset buttons, inline bounds validation, Reset button, and accessible labels. |
| `frontend/src/components/PredictionResultCard.tsx` | Unified result card featuring animated `ScoreGauge` SVG ring, Verdict pill, uncalibrated propensity progress bar, artifact/feature metadata grid, expandable **Leakage Governance** chip inspector (`11 consumed / 8 excluded`), explicit Offline Demo warning banner when disconnected, and `Ask Gemini Co-Pilot` quick link. |
| `frontend/src/pages/HomePage.tsx` | Hero workbench summary, 5-card KPI banner (`4,500 Claims`, `10 Models`, `100.00% Best ML PR-AUC`, `99.11% Best DL PR-AUC`, `29 -> 28 Contract`), clickable 3-approach cards, and a filterable **All 10 Models Benchmark Table** (`All / Classical ML / Deep Learning`) with live backend refresh (`Promise.allSettled`) + cached fallback + one-click `Test Model` jump actions. |
| `frontend/src/pages/ClassicalMLPage.tsx` | Approach 1 playground covering `XGBoost`, `RandomForest`, `GradientBoosting`, `SVM`, and `LogisticRegression` with test AP badges on model pills, `Predict a Claim` view, and `Benchmark Results (5 Models)` table. |
| `frontend/src/pages/DLPage.tsx` | Approach 2 playground covering `TabTransformer`, `CNN1D`, `ResidualMLP`, `MLP`, and `Autoencoder` (clearly badged as Negative Result at `MSE 0.1286`), paired with a complete `Benchmark Results (5 Architectures)` comparison tab. |
| `frontend/src/pages/AgenticPage.tsx` | Renders the 6-role `AgentCard` architecture grid (`Ingestion` through `Audit`), explicit **Design vs. Runtime** notice, preset-enabled 9-field + optional ICD/procedure code form, `ScoreGauge` output, timeline `<AuditTrail />`, and client-side `Export Audit JSON` download. |
| `frontend/src/lib/api.ts` | Centralized `fetchJsonWithTimeout` HTTP wrapper with `AbortController` timeouts (5s health, 10s metrics, 15s ML/DL, 20s agentic, 30s chat) and explicit `response.ok` error parsing. |

---

## 3. Node.js Express Gateway (`backend/node-api/src/`)

Built with Express + TypeScript (`npm run dev` via `tsx watch src/server.ts`, listening on `PORT` default `3000`):

* **`server.ts`:** Configures CORS (`cors()`), JSON body parsing, and mounts:
  * `/api/health` $\rightarrow$ `routes/health.ts` (`GET /` returns `{"status":"ok"}`)
  * `/api` $\rightarrow$ `routes/ml.ts`
  * `/api/chat` $\rightarrow$ `routes/chat.ts`
  * `/agentic` $\rightarrow$ `agentic/predict.ts`
* **`routes/ml.ts`:**
  * `POST /api/ml/predict?model=<name>` proxies request JSON cleanly to `${ML_SERVICE_URL}/predict/ml?model=...` (`127.0.0.1:8000` default) and returns HTTP `502` JSON with error detail if Python is offline.
  * `POST /api/dl/predict?model=<name>` proxies request JSON cleanly to `${ML_SERVICE_URL}/predict/dl?model=...`.
  * `GET /api/results/ml` reads `Approch-1/evaluation/model_results.json`.
  * `GET /api/results/dl?model=<name>` & `GET /api/results/dl/all` read consolidated JSON benchmarks across `mlp`, `residualmlp`, `cnn1d`, `tabtransformer`, and `autoencoder` inside `Approch-2-DL/results/`.
* **`agentic/predict.ts`:**
  * `POST /agentic/predict` validates `ClaimInput` (`claim_amount`, `patient_age`, `patient_income`, `claim_type`, `provider_specialty`, `cluster`, `submission_method`, optional `diagnosis_codes`, `procedure_codes`), computes `AgenticEnsemble-v1` score, and emits a 4-stage `audit_trail` (`ingestion`, `validation`, `risk-scorer`, `xai-explainer`).
* **`routes/chat.ts`:**
  * Uses `@google/generative-ai` (`GoogleGenerativeAI`) initialized with `process.env.GEMINI_API_KEY` targeting **`gemini-2.0-flash`**, forwarding user `message` and prior `history` with a healthcare fraud analyst `systemInstruction`.

---

## 4. Python FastAPI ML Inference Service (`ml-service/app/`)

Executed via `uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir ml-service`:

### 4.1 Strict Request/Response Contracts (`app/schemas/claims.py`)
`ClaimInput` enforces `ConfigDict(extra="forbid")` so undocumented or typoed keys fail fast with HTTP `422`:
* **11 Consumed Fields:** `ClaimAmount` ($\ge 0$), `ClaimDate` (`YYYY-MM-DD`, decomposed into `ClaimYear`, `ClaimMonth`, `ClaimDayOfWeek`), `PatientAge` (`0..120`), `PatientGender` (`"F" | "M"`), `ProviderSpecialty` (5 allowed categories), `PatientIncome` ($\ge 0$), `PatientMaritalStatus` (4 categories), `PatientEmploymentStatus` (4 categories), `ClaimType` (4 categories), `ClaimSubmissionMethod` (3 categories), `Cluster` (`0..3`).
* **Accepted but Strictly Excluded Metadata:** `ClaimID`, `PatientID`, `ProviderID`, `DiagnosisCode`, `ProcedureCode`, `ClaimStatus`, `ProviderLocation` are accepted for payload compatibility and explicitly excluded from model input vectors (`excluded_input_fields`).

### 4.2 Inference Services (`classical_service.py` & `dl_service.py`)
* **Artifact Caching (`@lru_cache`):** Loads `Approch-1/data/processed/preprocessor.joblib`, `.joblib` scikit-learn/XGBoost pipelines, and PyTorch `.pt` state dicts once in memory on CPU.
* **Shape & Finite Value Guards:** Verifies transformed numpy output equals `(1, 29)` float32 with zero `NaN`/`Inf` values prior to model execution.
* **Autoencoder Scoring Logic:** Runs `model.reconstruction_error(feat_tensor)` and compares against the fitted checkpoint `threshold` (`0.1286`); converts `(recon_error - threshold)` via sigmoid scaling so `>= 0.50` maps consistently to `"Fraud"` while preserving `score_is_calibrated = False`.
* **SVM Scoring Logic:** Uses `scipy.special.expit(model.decision_function(features)[0])` when `predict_proba` is unavailable or returns predicted probabilities directly when present.

---

## 5. Standalone Agentic XAI Scaffold (`Approach-3-Agentic-AI/`)

A dedicated FastAPI module designed for port `8020` (`Approach-3-Agentic-AI/app/main.py`):
* `POST /agentic/predict/`: Accepts `ClaimInput` plus optional `diagnosis_codes` and `procedure_codes`.
* `app/services/xai_service.py`: Provides `XAIExplainer` (directional heuristic importance mapping across `claim_amount`, `cluster`, `patient_age`, `patient_income`, plus fallback channels) and `log_audit()`, which stamps UTC ISO timestamps and truncates SHA-256 digests (`hashlib.sha256(...).hexdigest()[:12]`) into an in-memory `audit_log` list.
* **Architectural Note:** Unifying `:8000` model inference directly into `:8020` / `:3000` `/agentic/predict` with persistent storage and vector-DB policy RAG remains the next planned production enhancement beyond Mini-Project 2 scope.
