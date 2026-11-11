# Volume 11 — System & Model Cards: Approach 3 Agentic Chain, `XAIExplainer` & Gemini Copilot

> **Document Classification:** Standardized Operational System Cards (Workflow, Attribution, Audit & LLM Layers)  
> **Primary Ownership:** **B. Varshith (`23BDS011`)**  
> **Primary Ground-Truth Sources:** [`Approach-3-Agentic-AI/app/main.py`](../../Approach-3-Agentic-AI/app/main.py), [`app/api/predict.py`](../../Approach-3-Agentic-AI/app/api/predict.py), [`app/services/xai_service.py`](../../Approach-3-Agentic-AI/app/services/xai_service.py), [`run_agentic.py`](../../Approach-3-Agentic-AI/run_agentic.py), [`backend/node-api/src/agentic/predict.ts`](../../backend/node-api/src/agentic/predict.ts), [`backend/node-api/src/routes/chat.ts`](../../backend/node-api/src/routes/chat.ts), [`frontend/src/pages/AgenticPage.tsx`](../../frontend/src/pages/AgenticPage.tsx), and [`frontend/src/agentic/components/ChatBox.tsx`](../../frontend/src/agentic/components/ChatBox.tsx).

---

## 1. System Card Portfolio Summary (`Approach 3`)

Unlike Approaches 1 and 2 — which serialize empirical statistical models in `.joblib` and `.pt` files — Approach 3 is implemented as **four decoupled architectural capabilities** distributed across the standalone `:8020` Python prototype, the Express `:3000` gateway, and React `:5173`:

| System Card ID | Subsystem Component | Runtime Location | Active Integration Status | Calibration / Reliability Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **`CARD-A3-01`** | `AgenticEnsemble-v1` Risk Orchestrator | `node-api/src/agentic/predict.ts` (`:3000`) & `Approach-3/app/api/predict.py` (`:8020`) | Node `:3000` **live via UI**; `:8020` **standalone** (`run_agentic.py` broken) | **Uncalibrated Rule Stub** (saturates to `1.0` at `claim_amount >= \$260–\$300`) |
| **`CARD-A3-02`** | `XAIExplainer` Feature Attribution Module | `Approach-3/app/services/xai_service.py` & mirrored in `predict.ts` | Returns `"shap-approximation"` dictionary | **Static Constant Priors** (`0.30, 0.25, 0.15, 0.10, 0.05`) — not true SHAP |
| **`CARD-A3-03`** | SHA-256 `audit_log` Provenance Tracker | In-memory list (`xai_service.py`) + response payload hashes (`predict.ts`) | Active in runtime JSON payloads (`12`-char hex digest) | **Non-Persistent Volatile Memory** (resets on restart, unchained hashes) |
| **`CARD-A3-04`** | `gemini-3.8-flash` SIU Analyst Copilot | `backend/node-api/src/routes/chat.ts` (`:3000`) + `ChatBox.tsx` | **Mounted live** on React `agentic` tab | Context-grounded Generative QA (requires key in `Approach-3-Agentic-AI/.env`) |

---

## 2. System Card `CARD-A3-01`: `AgenticEnsemble-v1` Multi-Stage Chain

### 2.1 Card Metadata & Intended Role

* **Emitted Model Identity:** `model_name = "AgenticEnsemble-v1"`, `agent_id = "agent-chain-001"`, `feature_count = 9`
* **Execution Graph (Fixed 4-Stage Synchronous Sequence):**
  ```text
  ingestion  -->  validation  -->  risk-scorer  -->  xai-explainer
  ```
* **Intended Role:** Demonstrate end-to-end synchronous workflow telemetry, stage-by-stage timing/decision traceability, and structured audit packaging between claim ingestion and analyst presentation (`AgenticPage.tsx`).
* **Out-of-Scope Role (Strictly Enforced in Documentation):** Quantitative fraud risk adjudication against `Health Insurance Fraud Claims.xlsx` (`\$100.12–\$9,997.20`), because the implemented risk kernel is an uncalibrated linear clamp rather than an empirical model.

### 2.2 Exact Risk Scoring Mechanics & Dollar Calibration Audit

Both Python (`predict.py`) and TypeScript (`predict.ts`) implement the exact arithmetic:

$$s(\text{amt}, \text{age}) = \min\Big(\max\big(0.01 \cdot \text{claim\_amount} + 0.005 \cdot \text{patient\_age} - 2.0,\; 0.0\big),\; 1.0\Big)$$

#### Exact Piecewise Behavior Analysis

| Operational Region | Algebraic Condition | Emitted Continuous Score $s$ | Discrete Flag (`s >= 0.5`) | Clinical Dataset Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Lower Zero Floor** | $\text{amt} \le 200 - 0.5\cdot\text{age}$ | `0.0000` | Legitimate | Covers only trivial claims under `\$150–\$200` |
| **Sub-Threshold Benign** | $200 - 0.5\cdot\text{age} < \text{amt} < 250 - 0.5\cdot\text{age}$ | `(0.0, 0.5)` | Legitimate | Exact `\$50` dollar band below flip threshold |
| **Upper Pre-Ceiling Fraud** | $250 - 0.5\cdot\text{age} \le \text{amt} < 300 - 0.5\cdot\text{age}$ | `[0.5, 1.0)` | Fraud | Exact `\$50` dollar transition band above flip threshold |
| **Saturated Ceiling (`1.0`)** | $\text{amt} \ge 300 - 0.5\cdot\text{age}$ | **`1.0000`** *(clamped)* | **Fraud** | **Captures `100%` of realistic hospital claims above `\$300`!** |

Look closely at what happens across the canonical dataset's legitimate mean (`\$4,776.80`):

* For a 40-year-old claimant submitting an ordinary legitimate `\$4,776.80` claim (`y = 0`), raw inner term is `(4776.80 * 0.01) + (40 * 0.005) - 2.0 = 47.768 + 0.20 - 2.0 = +45.968`, clamped abruptly to **`1.0000` (100% Fraud)**!
* Thus `AgenticEnsemble-v1` exhibits near-100% false-positive alarm saturation on standard healthcare amounts above `\$300` while ignoring `ClaimType`, `ProviderSpecialty`, `ClaimSubmissionMethod`, `Cluster`, and `PatientIncome` inside its risk scalar calculation despite labeling `feature_count = 9`.

### 2.3 Code Defect Audit (`Approach-3-Agentic-AI/` `:8020` Prototype)

| Artifact | Verified Bug / Quirk | Exact Failure Mode | Remediation |
| :--- | :--- | :--- | :--- |
| `run_agentic.py` (`12` lines) | `from app.api.predict import predict_fraud` passed to `app.include_router(predict_fraud)` | Passing an `async def` view coroutine into `FastAPI.include_router()` throws runtime error (`predict_fraud` lacks `.routes`). | Export `router` from `predict.py` and call `app.include_router(predict_router)` or start `uvicorn app.main:app --port 8020`. |
| `app/api/predict.py` (`50` lines) | Imports `BackgroundTask` (invalid/singular) & defines `router = APIRouter(...)` twice (Lines `6` and `30`) | Line `30` overwrites Line `6`'s router reference silently, orphaning routes declared between lines `6–29`. | Remove duplicate `APIRouter` assignment on Line `30`; replace `BackgroundTask` with valid `BackgroundTasks` (or remove unused import). |
| `node-api/src/agentic/predict.ts` (`39` lines) | Duplicates heuristic inline inside Express rather than querying `:8000` or `:8020` | Causes `:8020` Python `XAIExplainer` changes not to reflect in live `:5173` UI traffic via `:3000`. | Call `POST :8000/predict/ml` (`RandomForest` / `XGBoost`) inside Node `predict.ts` risk-scorer stage. |

---

## 3. System Card `CARD-A3-02`: `XAIExplainer` Attribution Service

### 3.1 Card Metadata & Static Weight Specification

* **Source Module:** `XAIExplainer` class inside [`Approach-3-Agentic-AI/app/services/xai_service.py`](../../Approach-3-Agentic-AI/app/services/xai_service.py) (`43` lines).
* **Declared Protocol Field:** `method = "shap-approximation"`
* **Emitted Feature Weight Map:**
  * `claim_amount`: **`0.30`**
  * `cluster`: **`0.25`**
  * `patient_age`: **`0.15`**
  * `patient_income`: **`0.10`**
  * *All remaining features*: **`0.05`** fallback default

### 3.2 Formal Attribution Audit (`Static Prior` vs. `Local Shapley`)

| Shapley Property | Mathematical Requirement | Behavior of Current `XAIExplainer` | Architectural Impact |
| :--- | :--- | :--- | :--- |
| **Local Accuracy (Efficiency)** | $\sum_{j} \phi_j = f(\mathbf{x}) - \mathbb{E}[f(X)]$ | Sum is constant (`0.80 + 0.05*k`) independent of model prediction `0.0` vs `1.0`. | Bar chart heights remain frozen even when claim amounts drop by `10x`! |
| **Input Sensitivity** | Altering `ClaimAmount` changes $\phi_{\text{amount}}$ | `claim_amount` returns fixed `0.30` regardless of `\$100` vs `\$9,900`. | UI communicates global domain heuristic importance, not case-specific drivers. |
| **Null-Player Axiom** | Zero-gradient inputs satisfy $\phi_j = 0$ | `cluster` gets `0.25` and `patient_income` gets `0.10` despite `0` mathematical coefficient in `AgenticEnsemble-v1`. | Attributes synthetic risk weight to unused attributes under active heuristic mode. |

**Upgrade Blueprint:** Integrating `shap.TreeExplainer` over Approach 1's `XGBClassifier` (`85.8 KB`) or `HistGradientBoostingClassifier` (`72.4 KB`) inside `:8000` produces true additive per-claim contributions in milliseconds (`< 5 ms` per claim).

---

## 4. System Card `CARD-A3-03`: SHA-256 `audit_log` Provenance Layer

### 4.1 Mechanics & Durability Boundaries

* **Source Implementation:** `log_audit` function (`Approach-3-Agentic-AI/app/services/xai_service.py`) appending records into module-level list `audit_log`, mirrored functionally by per-step status emission in Node `predict.ts`.
* **Cryptographic Hash Format:** `hashlib.sha256(...).hexdigest()[:12]` yielding compact `12`-character hex tags (`48` bits of entropy, collision probability ~$50\%$ after $\sim 2^{24} \approx 16.7$ million claims — adequate as request reference tags, shorter than full `64`-char SHA-256 digests).
* **Operational Tradeoffs:**
  1. **Volatile RAM Lifetime:** Stopping or restarting Uvicorn `:8020` clears `audit_log` completely.
  2. **Single-Worker Assumption:** Running multiple Uvicorn `--workers > 1` isolates separate Python memory spaces so `GET /api/v1/audit` only observes events scored by the single worker handling that GET request.
  3. **Absence of Chained Sequencing:** Independent hashes detect accidental corruption of an entry's payload, but lack prior-hash chaining (`prev_hash`) needed to prove log append-only integrity against deletion/reordering.

---

## 5. System Card `CARD-A3-04`: `gemini-3.8-flash` Analyst Copilot

### 5.1 Card Identity, Endpoint Binding & Prompt Grounding

* **Backend Handler:** `backend/node-api/src/routes/chat.ts` (`47` lines) handling `POST /api/chat`.
* **Verified LLM Identifier:** **`"gemini-3.8-flash"`** instantiated via `@google/generative-ai` (`GoogleGenerativeAI`) — correcting prior markdown docs that stated `gemini-2.0-flash`.
* **Credential Resolution:** Invokes `dotenv.config(...)` pointing across sibling workspace boundaries at **`../../Approach-3-Agentic-AI/.env`** for `GEMINI_API_KEY`.
* **Frontend Mount Scope:** [`frontend/src/agentic/components/ChatBox.tsx`](../../frontend/src/agentic/components/ChatBox.tsx) (`309` lines) renders strictly when `activeTab === "agentic"` inside `App.tsx` (`63` lines).

### 5.2 Copilot Failure Modes & Safety Guardrails

| Risk Vector | Current Mitigation / State | Production Requirement |
| :--- | :--- | :--- |
| **Missing `GEMINI_API_KEY` in Prototype `.env`** | `Approach-3-Agentic-AI/.env` absent or invalid triggers runtime error inside `chat.ts` when handling `/api/chat`. | Fail fast at Node `:3000` boot with explicit degraded status flag on `/api/health` + move secret into `backend/node-api/.env`. |
| **Hallucination / Grounding Drift** | Context prompt forwards claim inputs & prediction JSON from frontend state (`ChatBox.tsx`) alongside user query. | Ground prompt strictly on Tier 1 ML probabilities + true `TreeSHAP` values rather than stubbed `AgenticEnsemble-v1` `1.0` clamp outputs. |
| **PHI / PII Leakage Outside Premise** | Raw claim payloads pass over public Google Generative AI API; workbook lacks patient real names (synthetic `PatientID` dropped). | Enforce strip/redaction middleware on `PatientID`/`ClaimID` before dispatching prompts externally in real payer deployments. |
