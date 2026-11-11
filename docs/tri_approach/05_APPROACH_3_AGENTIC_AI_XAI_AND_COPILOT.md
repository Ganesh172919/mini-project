# Volume 05 — Approach 3: Agentic AI Orchestration, Explainability (`XAI`) & Gemini Copilot

> **Document Classification:** Subsystem Audit, Behavioral Proof & Integration Blueprint  
> **Primary Ownership:** **B. Varshith (`23BDS011`)**  
> **Distributed Across Three Codebase Regions:** [`Approach-3-Agentic-AI/`](../../Approach-3-Agentic-AI/) (`:8020` FastAPI prototype), [`backend/node-api/src/agentic/predict.ts`](../../backend/node-api/src/agentic/predict.ts) + [`routes/chat.ts`](../../backend/node-api/src/routes/chat.ts) (`:3000` Node gateway), and [`frontend/src/pages/AgenticPage.tsx`](../../frontend/src/pages/AgenticPage.tsx) + [`frontend/src/agentic/`](../../frontend/src/agentic/) + [`ChatBox.tsx`](../../frontend/src/agentic/components/ChatBox.tsx) (`:5173` React SPA).  
> **Primary Ground-Truth Sources:** [`Approach-3-Agentic-AI/app/main.py`](../../Approach-3-Agentic-AI/app/main.py), [`app/api/predict.py`](../../Approach-3-Agentic-AI/app/api/predict.py), [`app/services/xai_service.py`](../../Approach-3-Agentic-AI/app/services/xai_service.py), [`run_agentic.py`](../../Approach-3-Agentic-AI/run_agentic.py), [`docs/README.md`](../../Approach-3-Agentic-AI/docs/README.md), [`backend/node-api/src/agentic/predict.ts`](../../backend/node-api/src/agentic/predict.ts), [`backend/node-api/src/routes/chat.ts`](../../backend/node-api/src/routes/chat.ts), [`frontend/src/App.tsx`](../../frontend/src/App.tsx), [`frontend/src/pages/AgenticPage.tsx`](../../frontend/src/pages/AgenticPage.tsx), and [`frontend/src/agentic/`](../../frontend/src/agentic/data/agents.ts).

---

## 1. Strategic Mandate & Architectural Reality Check

Approaches 1 and 2 answer a narrow mathematical question: *given a pre-aligned 29-feature claim vector $\mathbf{x} \in \mathbb{R}^{29}$, what scalar fraud score $\hat{y} \in [0, 1]$ minimizes imbalanced classification risk?* While necessary, a bare scalar leaves healthcare Special Investigations Units (SIU) without workflow provenance, regulatory rationales, audit traceability, or natural-language investigation triage.

**Approach 3 addresses workflow orchestration, transparency, and investigator assistance** through four functional capabilities:

1. **Multi-Stage Pipeline Telemetry:** Modeling each prediction as a traceable chain of discrete agent states (`ingestion` → `validation` → `risk-scorer` → `xai-explainer`).
2. **Feature Attribution Explanations (`XAIExplainer`):** Returning structured factor breakdowns alongside binary fraud classifications (`"shap-approximation"` method label).
3. **SHA-256 Digest Audit Registry (`audit_log`):** Issuing deterministic `12`-character hex identifiers for claim screening trace records.
4. **LLM-Backed Analyst Copilot (`gemini-3.8-flash`):** Embedding contextual Q&A (`ChatBox.tsx` → `/api/chat`) aware of claim form parameters and current risk predictions.

### 1.1 Critical Engineering Distinction: Runtime Architecture vs. Empirical Models

Unlike Approaches 1 and 2 — which load fitted `.joblib` estimators and `.pt` PyTorch weights trained over all `4,500` records of `Health Insurance Fraud Claims.xlsx` — **Approach 3 currently runs an uncoupled analytical heuristic engine (`AgenticEnsemble-v1`) rather than calling trained Approach 1 or Approach 2 artifacts inside its orchestrator chain**. Furthermore, its Python implementation (`:8020`) and its active React-facing Node implementation (`:3000`) exhibit subtle divergence and three verifiable code defects documented transparently below so both evaluators and maintainers operate with exact ground truth.

---

## 2. Distributed Codebase Inventory Across Ports `:8020`, `:3000`, and `:5173`

```text
+---------------------------------------------------------------------------------------------------+
|                   DISTRIBUTED CODEBASE REALIZATION OF APPROACH 3 ACROSS 3 LAYERS                   |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Standalone Python Prototype :8020]       [Node Gateway Active Plane :3000]   [React UI :5173]   |
|  Approach-3-Agentic-AI/                    backend/node-api/                   frontend/src/      |
|  |-- app/main.py (24L)                     |-- src/agentic/predict.ts (39L)    |-- MOUNTED LIVE:  |
|  |-- app/api/predict.py (50L) [BUGS!]      |   Handles POST /api/agentic/      |   App.tsx ->     |
|  |-- app/services/xai_service.py (43L)     |   predict IN-PROCESS locally      |   AgenticPage    |
|  |-- run_agentic.py (12L)     [BROKEN!]    |   (does NOT proxy to :8020!)      |   + ChatBox      |
|  |-- requirements.txt (4L)                 |-- src/routes/chat.ts (47L)        |-- ON-DISK ONLY:  |
|  `-- docs/README.md (25L)                  |   Reads ../../Approach-3-         |   src/agentic/   |
|                                            |   Agentic-AI/.env -> calls        |   AgenticDash    |
|                                            |   gemini-3.8-flash                |   ClaimInputForm |
|                                                                                |   AuditTrail     |
|                                                                                |   AgentCard      |
|                                                                                |   ScoreGauge     |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 File-by-File Inventory Table

| File Path | Lines | Runtime Status | Core Responsibility |
| :--- | :---: | :--- | :--- |
| `Approach-3-Agentic-AI/app/main.py` | `24` | Standalone ASGI `:8020` | Initializes FastAPI app, registers CORS & `/api/v1` router, exposes root/health endpoints |
| `Approach-3-Agentic-AI/app/api/predict.py` | `50` | Standalone `:8020` **(defects)** | Defines heuristic risk score + `XAIExplainer` invocation + `audit_log` endpoints (`duplicate router`, invalid `BackgroundTask` import) |
| `Approach-3-Agentic-AI/app/services/xai_service.py` | `43` | Standalone `:8020` | Implements `XAIExplainer` static weight dictionary (`"shap-approximation"`) & `log_audit` SHA-256 registry |
| `Approach-3-Agentic-AI/run_agentic.py` | `12` | **Broken entrypoint** | Incorrectly imports `predict_fraud` handler and passes it to `app.include_router(predict_fraud)` |
| `Approach-3-Agentic-AI/docs/README.md` | `25` | Prototype docs | Documents conceptual standalone microservice design |
| `backend/node-api/src/agentic/predict.ts` | `39` | **Active Node runtime (`:3000`)** | Powers `POST /api/agentic/predict` directly in Node; runs matching heuristic + 4-stage audit trace + `AgenticEnsemble-v1` |
| `backend/node-api/src/routes/chat.ts` | `47` | **Active Node runtime (`:3000`)** | Powers `POST /api/chat`; loads `.env` from `../../Approach-3-Agentic-AI/.env` & queries `gemini-3.8-flash` |
| `frontend/src/pages/AgenticPage.tsx` | `247` | **Mounted in `App.tsx`** | Active UI panel on `"agentic"` tab; submits claim form to `/api/agentic/predict` and renders 4-step pipeline & attribution |
| `frontend/src/agentic/components/ChatBox.tsx` | `309` | **Mounted in `App.tsx`** (`agentic` tab) | Interactive floating/docked AI copilot drawer wired to `/api/chat` |
| `frontend/src/agentic/data/agents.ts` | `56` | Data catalog | Stores `6` conceptual specialist agent definitions/personas |
| `frontend/src/agentic/pages/AgenticDashboard.tsx` | `142` | **Unmounted sibling** | Alternative standalone agentic dashboard on disk (not imported by `App.tsx`) |
| `frontend/src/agentic/pages/ClaimInputForm.tsx` | `260` | **Unmounted sibling** | Standalone claim intake form component under `src/agentic/pages/` |
| `frontend/src/agentic/components/AuditTrail.tsx` | `157` | **Unmounted sibling** | Standalone expandable trace timeline widget on disk |
| `frontend/src/agentic/components/AgentCard.tsx`, `ScoreGauge.tsx` | — | **Unmounted siblings** | Modular visualization cards not referenced by `AgenticPage.tsx` or `App.tsx` |

---

## 3. Conceptual Vision vs. Actual Runtime Pipeline Execution

Comparing static UI documentation and [`frontend/src/agentic/data/agents.ts`](../../frontend/src/agentic/data/agents.ts) against the actual execution arrays emitted by `predict.ts` (`:3000`) and `predict.py` (`:8020`) reveals a two-tier maturity gap:

### 3.1 Conceptual `6`-Role Cognitive Agents vs. Executed `4`-Stage Runtime Trace

While `frontend/src/agentic/data/agents.ts` articulates six conceptual cognitive worker personas across specialized domain duties (such as intake parsing, demographic profiling, pattern matching, threshold adjudication, explainability generation, and compliance auditing), every live request to `POST /api/agentic/predict` emits a fixed synchronous **4-stage trace sequence** (`agent_id: "agent-chain-001"`):

```text
[Stage 1: ingestion]  -->  [Stage 2: validation]  -->  [Stage 3: risk-scorer]  -->  [Stage 4: xai-explainer]
 Parses raw claim          Checks schema presence      Evaluates linear clamp        Attaches static feature
 fields (`claim_amount`,   & boundary defaults;        formula `AgenticEnsemble-v1`   weights dictionary
 `patient_age`, etc.)      sets `feature_count=9`      into [0.0, 1.0]                + 12-char SHA-256 id
```

### 3.2 Request Execution Metadata Emitted by Active Pipeline (`predict.ts` / `predict.py`)

| Metadata Property | Exact Emitted Value | Architectural Reality |
| :--- | :--- | :--- |
| **`model_name`** | `"AgenticEnsemble-v1"` | Single deterministic TypeScript/Python rule formula; **not** an ensemble over `Approch-1` / `Approch-2-DL` weights |
| **`agent_id`** | `"agent-chain-001"` | Static chain identifier logged with pipeline response |
| **`feature_count`** | `9` | Distinct from the canonical `29`-feature preprocessing contract used in Approaches 1 and 2 |
| **Audit Stage Keys** | `["ingestion", "validation", "risk-scorer", "xai-explainer"]` | Synchronous deterministic sequence emitted per request (no asynchronous agent message passing or retry loops) |

---

## 4. Exact Risk Scoring Formula & Closed-Form Saturation Proof

Both `Approach-3-Agentic-AI/app/api/predict.py` and `backend/node-api/src/agentic/predict.ts` evaluate the exact identical affine clamped rule:

$$s_{\text{agentic}}(\text{amt}, \text{age}) = \min\Big(\max\big(0.01 \cdot \text{claim\_amount} + 0.005 \cdot \text{patient\_age} - 2.0,\; 0.0\big),\; 1.0\Big)$$

We now derive its exact operating regimes algebraically and compare them against the empirical dollar bounds of `Health Insurance Fraud Claims.xlsx` (`dataset_audit.json`).

### 4.1 Closed-Form Boundary & Saturation Equations

1. **Lower Floor Regime ($s_{\text{agentic}} = 0.0$):**
   $$0.01 \cdot \text{amt} + 0.005 \cdot \text{age} - 2.0 \le 0 \iff \text{amt} \le 200 - 0.5 \cdot \text{age}$$
2. **Active Linear Transition Band ($0 < s_{\text{agentic}} < 1$):**
   Only claims spanning a **exact `\$100.00` dollar window** produce non-saturated continuous scores! Inside this window:
   * **Binary Fraud Flip Boundary ($s_{\text{agentic}} = 0.50$):**
     $$0.01 \cdot \text{amt} + 0.005 \cdot \text{age} - 2.0 = 0.5 \iff \text{amt}_{0.50} = 250 - 0.5 \cdot \text{age}$$
     Notice the age slope equivalence: `20` extra years of age shift the threshold downward by merely `\$10.00` (`0.005 * 20 = 0.10` score increase).
3. **Ceiling Saturation Regime ($s_{\text{agentic}} = 1.00$):**
   $$0.01 \cdot \text{amt} + 0.005 \cdot \text{age} - 2.0 \ge 1.0 \iff \text{amt}_{\text{sat}} \ge 300 - 0.5 \cdot \text{age}$$

### 4.2 Sensitivity Table Across Representative Claimant Ages vs. Empirical Dataset Dollars

| Claimant Age (`years`) | Zero Clamp (`score = 0.0`) | Decision Flip (`score = 0.50`) | Ceiling Saturation (`score = 1.00`) | Score on Workbook Min Legit (`\$100.12`) | Score on Workbook Mean Legit (`\$4,776.80`) | Score on Workbook Min Fraud (`\$7,465.07`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`20`** | $\le \$190.00$ | $\$240.00$ | $\ge \$290.00$ | `0.0000` | **`1.0000` (Saturated FP)** | `1.0000` |
| **`40`** | $\le \$180.00$ | $\$230.00$ | $\ge \$280.00$ | `0.0000` | **`1.0000` (Saturated FP)** | `1.0000` |
| **`60`** | $\le \$170.00$ | $\$220.00$ | $\ge \$270.00$ | `0.0000` | **`1.0000` (Saturated FP)** | `1.0000` |
| **`80`** | $\le \$160.00$ | $\$210.00$ | $\ge \$260.00$ | `0.0000` | **`1.0000` (Saturated FP)** | `1.0000` |

#### Diagnostic Impact on Real Dataset Distribution

Examine why this calibration gap breaks empirical parity completely on `Health Insurance Fraud Claims.xlsx`:

* In the real workbook (`dataset_audit.json`), **all `270` true frauds** have `ClaimAmount >= \$7,465.07`, while **legitimate claims span `\$100.12` to `\$9,997.20`** (`mean = \$4,776.80`).
* Because ceiling saturation occurs above `\$260–\$300` (roughly **`\$250`–`\$300`**, over **`15×` smaller** than the legitimate mean of `\$4,776.80`!), `AgenticEnsemble-v1` clamps almost every ordinary claim cleanly onto **`1.0000` (Fraud)** unless `ClaimAmount` is a tiny sub-`\$250` minor charge! Notice too that earlier `ClaimAmount` coefficient `0.01` is literally missing two decimal zeros relative to dataset dollar scaling (`0.0001` rather than `0.01` would place the transition window at `\$20,000–\$30,000`, or subtracting `80` with `0.01` would target `\$8,000`).

---

## 5. `XAIExplainer` Attribution Mechanics vs. Formal Shapley Axioms

In [`Approach-3-Agentic-AI/app/services/xai_service.py`](../../Approach-3-Agentic-AI/app/services/xai_service.py) (`43` lines), class `XAIExplainer` constructs feature importance dictionaries tagged `method: "shap-approximation"` for downstream UI bars and analyst explanations.

### 5.1 Static Attribution Dictionary Audit

Directly inspecting `XAIExplainer` reveals constant importance weights keyed by canonical attribute name:

| Attribute Key | Static Weight Returned by `XAIExplainer` | Share Across Named Weights | Dependence on Current Claim Input Value? |
| :--- | :---: | :---: | :--- |
| **`claim_amount`** | `0.30` | Largest constant | **No** — returns `0.30` whether `ClaimAmount` is `\$120` or `\$9,950` |
| **`cluster`** | `0.25` | 2nd constant | **No** — returns `0.25` regardless of `Cluster` (`0`, `1`, `2`, `3`) |
| **`patient_age`** | `0.15` | 3rd constant | **No** — invariant to claimant age |
| **`patient_income`** | `0.10` | 4th constant | **No** — invariant to income |
| *Other default features* | `0.05` (each) | Fallback default | **No** — constant fallback weight |

### 5.2 Violations of Cooperative Game-Theoretic Shapley Axioms

True SHAP (SHapley Additive exPlanations) values $\phi_j(f, \mathbf{x})$ uniquely satisfy four axioms for any scoring function $f(\mathbf{x})$ and baseline expectation $\phi_0 = \mathbb{E}[f(X)]$:

1. **Local Accuracy (Efficiency Violation):**
   $$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{M} \phi_j(f, \mathbf{x})$$
   Because `XAIExplainer` returns static weights `{0.30, 0.25, 0.15, 0.10, ...}` irrespective of whether $f(\mathbf{x}) = 0.0$ or $f(\mathbf{x}) = 1.0$, $\sum_j \phi_j$ never reconciles additively with $f(\mathbf{x}) - \phi_0$.
2. **Dummy / Null-Player Violation:**
   If `AgenticEnsemble-v1` strictly computes `0.01*amt + 0.005*age - 2.0`, features such as `cluster` and `patient_income` have **zero mathematical partial derivative** ($\partial f / \partial \text{cluster} = 0$, $\partial f / \partial \text{income} = 0$) and must carry exact Shapley value $\phi_{\text{cluster}} = \phi_{\text{income}} = 0.0$. Yet `XAIExplainer` awards `0.25` (`cluster`) and `0.10` (`patient_income`), creating non-zero attributions for null features.
3. **Symmetry & Saturation Flatness Violation:**
   Inside the upper saturation plateau (`claim_amount > \$300`), local gradient sensitivity across small claim changes is identically zero until dropping back toward the boundary; static positive attribution masks saturation completely from analysts.

**Documentation Standard:** Accordingly, this repository labels `XAIExplainer` outputs transparently as **static heuristic factor priors** inspired by domain importance (`ClaimAmount`, `Cluster`, `PatientAge`, `PatientIncome`) rather than true game-theoretic Shapley attributions (`shap.TreeExplainer` over Approach 1 XGBoost/RF).

---

## 6. In-Memory SHA-256 Audit Ledger (`audit_log`)

Inside [`Approach-3-Agentic-AI/app/services/xai_service.py`](../../Approach-3-Agentic-AI/app/services/xai_service.py), `log_audit` records claim evaluation events with compact cryptographic digests:

* **ID Generation:** Computes SHA-256 hash (`hashlib.sha256(...).hexdigest()[:12]`) producing a `12`-character (`48`-bit) hexadecimal `id`.
* **Storage Semantics:** Writes into an **in-memory Python list `audit_log`** within process RAM.
* **Production Readiness Constraints:**
  1. **Ephemeral State:** Restarting `:8020` clears `audit_log` cleanly to `[]` (no SQLite, PostgreSQL, or append-only WAL file persistence).
  2. **Non-Chained Digests:** Entries store independent `12`-char SHA-256 hashes without Merkle or previous-hash linking (`prev_hash`), providing convenient correlation IDs rather than cryptographic tamper-evidence.
  3. **Concurrency Limitation:** `list.append` under multi-worker ASGI (`--workers N > 1`) isolates per-worker memory spaces, causing `GET /api/v1/audit` to return partial logs across forked worker processes.

---

## 7. Verified Defects in the `:8020` Standalone Prototype (`Approach-3-Agentic-AI/`)

A line-level static audit of [`Approach-3-Agentic-AI/run_agentic.py`](../../Approach-3-Agentic-AI/run_agentic.py) (`12` lines) and [`app/api/predict.py`](../../Approach-3-Agentic-AI/app/api/predict.py) (`50` lines) pinpoints **three concrete bugs**:

| Defect ID | File & Location | Exact Code Issue Observed | Runtime Symptom & Workaround |
| :--- | :--- | :--- | :--- |
| **A3-BUG-01** | `Approach-3-Agentic-AI/run_agentic.py` (`1–12`) | Imports `predict_fraud` (async handler function) instead of an `APIRouter` instance and executes `app.include_router(predict_fraud)`. | Executing `python run_agentic.py` fails with router registration type error (`function` lacks `.routes`). **Workaround:** Start via `uvicorn app.main:app --port 8020`. |
| **A3-BUG-02** | `Approach-3-Agentic-AI/app/api/predict.py` | Imports invalid/non-existent symbol `BackgroundTask` instead of Starlette/FastAPI `BackgroundTasks`. | Can trigger `ImportError` on module import depending on FastAPI/Starlette export version! (`uvicorn` on certain environments fails if unresolved.) |
| **A3-BUG-03** | `Approach-3-Agentic-AI/app/api/predict.py` (Lines `6` & `30`) | Re-assigns `router = APIRouter(...)` a second time on line `30` after already instantiating `router = APIRouter(...)` on line `6`. | Any decorator routes bound to the line `6` `router` object before line `30` are silently overwritten/orphaned when `main.py` imports `predict.router`! |

---

## 8. Gemini Conversational Copilot (`backend/node-api/src/routes/chat.ts` & `ChatBox.tsx`)

The fourth active component of Approach 3 couples the React frontend to Google's Generative AI SDK inside Express Tier 2 (`POST /api/chat`).

### 8.1 Express Backend Route Architecture (`chat.ts`, `47` lines)

1. **Credential Path Coupling:** `chat.ts` loads `dotenv` from the sibling prototype directory relative path:
   ```text
   ../../Approach-3-Agentic-AI/.env
   ```
   looking up `GEMINI_API_KEY` at process startup. Notice why this coupling matters operationally: even though `node-api` does not call Python `:8020` at runtime, **starting `/api/chat` reads configuration secrets out of the sibling `Approach-3-Agentic-AI/.env` folder**.
2. **SDK & Exact Model Version (`gemini-3.8-flash`):** Line `13` of `backend/node-api/src/routes/chat.ts` instantiates `GoogleGenerativeAI` configuring model **`"gemini-3.8-flash"`** (correcting stale external summaries that wrote `gemini-2.0-flash`).
3. **Context Injection Flow:** Combines analyst questions (`message`) with serialized current claim attributes and active prediction results (`context`) into the prompt sent to `generateContent(...)`, returning structured explanatory prose summarizing risk indicators for SIU reviewers.

### 8.2 React Conditional Mounting (`App.tsx` vs. `frontend/src/agentic/`)

Inspecting [`frontend/src/App.tsx`](../../frontend/src/App.tsx) (`63` lines) verifies the exact live UI boundaries:

* **When `ChatBox` Mounts:** `ChatBox.tsx` (`309` lines) is mounted exclusively when `activeTab === "agentic"` inside `App.tsx`, hiding the chat drawer on `Overview`, `Classical ML`, and `Deep Learning` views.
* **Active UI vs. Unmounted Modular Components:**

```text
frontend/src/
|-- App.tsx (63L) ------------------------> MOUNTS:
|                                            - HomePage.tsx
|                                            - ClassicalMLPage.tsx
|                                            - DLPage.tsx
|                                            - AgenticPage.tsx (247L) [activeTab === "agentic"]
|                                            - ChatBox.tsx (309L)     [activeTab === "agentic"]
`-- agentic/
    |-- data/agents.ts (56L)                <-- Shared 6-agent metadata descriptors
    |-- pages/AgenticDashboard.tsx (142L)   <-- NOT imported by App.tsx
    |-- pages/ClaimInputForm.tsx (260L)     <-- NOT imported by App.tsx
    `-- components/
        |-- AuditTrail.tsx (157L)           <-- NOT imported by App.tsx
        |-- AgentCard.tsx                   <-- NOT imported by App.tsx
        `-- ScoreGauge.tsx                  <-- NOT imported by App.tsx
```

Notice why knowing this exact distinction saves developer hours: modifying `AgenticDashboard.tsx` or `AuditTrail.tsx` under `frontend/src/agentic/` will **not** alter the live UI at `http://localhost:5173` unless wired explicitly into `App.tsx` or imported into `AgenticPage.tsx` (`247` lines).

---

## 9. Production Unification & Remediation Roadmap for Approach 3

To elevate Approach 3 from a decoupled workflow & UX prototype into an empirically grounded enterprise orchestrator without discarding its modular React components and Gemini copilot UX:

1. **Replace `AgenticEnsemble-v1` Stub Formula with Real Trained Models:** Configure Node `predict.ts` (and Python `:8020` `predict.py`) `risk-scorer` agent stage to invoke Tier 1 FastAPI `POST /predict/ml` (e.g., Track B's selected `RandomForest` `intake` estimator or Track A `XGBoost.joblib`) rather than `0.01*amt + 0.005*age - 2.0`. This instantly removes the `\$250–\$300` saturation defect while preserving exact `4`-stage orchestration traces.
2. **Compute Dynamic Local Attributions (`TreeSHAP` / Leave-One-Out Sensitivity):** Replace static weights `{claim_amount: 0.3, cluster: 0.25, ...}` in `xai_service.py` with input-dependent `shap.TreeExplainer` values for tree models or gradient×input / perturbation attribution deltas for PyTorch models, satisfying Shapley Local Accuracy.
3. **Repair `:8020` Router Bugs & Background Import:** Fix `BackgroundTask` → `BackgroundTasks` and eliminate the shadowed second `router = APIRouter(...)` in `Approach-3-Agentic-AI/app/api/predict.py` L6/L30, and fix `run_agentic.py` to import `predict.router` cleanly.
4. **Persist `audit_log` to Durable Append-Only Storage:** Replace volatile RAM lists with SQLite / JSONL append-only cryptographic hash chains (`prev_hash` + full `64`-char SHA-256 digest) keyed across distributed workers.
5. **Reconcile `AgenticPage.tsx` vs. `src/agentic/pages/AgenticDashboard.tsx`:** Either compose the clean modular cards (`AgentCard`, `AuditTrail`, `ScoreGauge`, `ClaimInputForm`) inside `AgenticPage.tsx` or archive duplicate standalone drafts cleanly.
