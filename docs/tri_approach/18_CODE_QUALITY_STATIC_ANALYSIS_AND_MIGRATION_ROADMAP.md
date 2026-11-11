# Volume 18 — Static Code Quality Audit, Technical Debt Ledger & Migration Roadmap

> **Document Classification:** Static Code Review, Defect Remediation & MLOps Evolution Architecture  
> **Scope:** Consolidates every verified bug, configuration smell, architectural coupling, and documentation mismatch discovered across Python, TypeScript, React, and PowerShell files, paired with a phased modernization plan.  
> **Primary Ground-Truth Sources:** Direct line-level inspection across `Approch-1/`, `Approch-2-DL/`, `Approach-3-Agentic-AI/`, `ml-service/`, `backend/node-api/`, `frontend/src/`, `submission/`, and `docs/`.

---

## 1. Consolidated Code-Level Defect & Smell Registry

Every technical defect verified directly inside source files across the workspace is categorized below by severity, exact file path, verified code pattern, and clean remediation patch design:

| Issue ID | Severity | File & Verified Location | Exact Code Issue Observed | Impact / Failure Trigger | Clean Engineering Fix |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **`DEBT-01`** | **High** | `Approach-3-Agentic-AI/run_agentic.py` (`1–12`) | Imports `predict_fraud` handler coroutine from `app.api.predict` and passes it to `app.include_router(predict_fraud)`. | Launching `python run_agentic.py` crashes during ASGI boot because a view function is not an `APIRouter`. | Export unified `router` in `predict.py` and pass `app.include_router(router)` or run `uvicorn app.main:app --port 8020`. |
| **`DEBT-02`** | **High** | `Approach-3-Agentic-AI/app/api/predict.py` (`L6, L30`, import block) | Imports non-standard `BackgroundTask` (singular) instead of `BackgroundTasks` **and** instantiates `router = APIRouter(...)` twice (Lines `6` & `30`). | Line `30` silently shadows/overwrites the Line `6` router instance, dropping any endpoint bound between lines `6–29`; invalid import risks `ImportError`. | Remove duplicate `router = ...` declaration on Line `30` and fix or remove unused `BackgroundTask` import. |
| **`DEBT-03`** | **High** | `backend/node-api/src/agentic/predict.ts` (`39L`) & `Approach-3-Agentic-AI/app/api/predict.py` | Evaluates rule `min(max(0.01*amt + 0.005*age - 2.0, 0), 1)` instead of trained Approach 1/2 models; labels `"AgenticEnsemble-v1"` & `feature_count=9`. | Clamps virtually all legitimate claims above `\$300` (workbook legitimate mean `\$4,776.80`) to `1.00` Fraud! | Proxy `risk-scorer` stage to Tier 1 `POST :8000/predict/ml` (`XGBoost` or Track B `RandomForest` `intake`), retaining the 4-stage audit envelope. |
| **`DEBT-04`** | **Medium** | `frontend/src/pages/ClassicalMLPage.tsx` (`L92–98`) | Offline client fallback formula `clamp(ClaimAmount*0.00003 + PatientAge*0.001 - 1.5, 0.01, 0.99)`. | Clamps every claim under `\$47,033.33` (including `100%` of workbook claims `<= \$9,997.20`) identically to `0.01` when `:3000`/`:8000` is stopped. | Show unmistakable `"BACKEND OFFLINE — PREVIEW STUB ONLY"` modal banner and align coefficients/domain threshold cleanly if offline simulation remains. |
| **`DEBT-05`** | **Medium** | `Approch-1/scripts/create_notebooks.py` (`1–32`) | Code-generation template leaves `{class_def}` unformatted inside template string. | Script cannot programmatically generate executable Jupyter notebooks cleanly. | Supply missing `class_def` template dictionary or replace script with static curated `.ipynb` artifacts. |
| **`DEBT-06`** | **Medium** | `Approch-1/scripts/{preprocess,train_all_models}.py` | Expects paths prefixed with `project/` inside current working directory (`Approch-1/project/...`). | Raises `FileNotFoundError` on clean git clones without Windows NTFS directory junction. | Support `Path(__file__).resolve().parent.parent` path discovery alongside junction backwards-compatibility (`ADR-01`). |
| **`DEBT-07`** | **Medium** | `Approach-3-Agentic-AI/app/services/xai_service.py` (`43L`) | `XAIExplainer` returns static weights (`claim_amount:0.3, cluster:0.25, patient_age:0.15, patient_income:0.1, default:0.05`) + volatile Python RAM list `audit_log`. | Attribution chart is input-invariant (fails Shapley Local Accuracy); audit trail disappears on Uvicorn restart or splits across multi-worker forks. | Compute `shap.TreeExplainer` over tree weights + persist hash-chained (`prev_hash` + 64-char SHA-256) audit entries to SQLite WAL. |
| **`DEBT-08`** | **Low-Med** | `backend/node-api/src/routes/chat.ts` (`47L`) | Reaches across workspace boundary (`../../Approach-3-Agentic-AI/.env`) to locate `GEMINI_API_KEY`. | Breaks clean containerized deployment of `backend/node-api` unless `Approach-3-Agentic-AI/` volume is also mounted. | Configure standard `dotenv` search inside `backend/node-api/.env` + `process.env.GEMINI_API_KEY`. |
| **`DEBT-09`** | **Low-Med** | `frontend/src/App.tsx` vs `src/agentic/` & `src/components/` | `AgenticDashboard`, `ClaimInputForm`, `AuditTrail`, `AgentCard`, `ScoreGauge`, `PipelinePreview`, `ThemeToggle`, `useApiHealth` remain unmounted on disk. | Confuses frontend contributors and duplicates claim-input / gauge / audit UX effort. | Consolidate into one cohesive component tree mounting `ScoreGauge`, `ThemeToggle`, and `useApiHealth` inside `App.tsx`. |
| **`DEBT-10`** | **Low** | `Approch-1/models/SVM.joblib` | Fitted with `probability=True` (`predict()` flags `51`, `predict_proba>=0.5` flags `39`). | Creates mismatch between offline F1 logs (`83.52%`) and live `/predict/ml` API decisions (`91.14%` F1). | Already cleanly mitigated in Track B (`SVC(probability=False)` + `expit`); document prominently on Classical ML UI. |

---

## 2. Target Architecture Migration Roadmap (`Phase 1` → `Phase 3`)

Moving cleanly from academic dual-track research toward hardened enterprise production executes across three low-risk incremental phases:

### Phase 1 — Immediate Non-Breaking Integration Fixes (`P0`)

1. **Unify Agentic Scoring with Tier 1 Empirical ML:** Update `backend/node-api/src/agentic/predict.ts` so its `risk-scorer` step executes an internal HTTP POST to `http://127.0.0.1:8000/predict/ml` (selecting `XGBoost` or Track B's `intake` `RandomForest` model), transforming inbound claim attributes to strict PascalCase `ClaimRequest` keys cleanly.
2. **Repair Prototype `:8020` FastAPI Module:** Deduplicate `router = APIRouter(...)` in `Approach-3-Agentic-AI/app/api/predict.py`, replace `BackgroundTask` with `BackgroundTasks`, and point `run_agentic.py` cleanly to `predict.router`.
3. **Decouple `.env` Secrets:** Load `.env` locally within `backend/node-api/` (falling back cleanly to environment injection for Docker/cloud targets) while retaining optional dev lookup backwards compatibility.

### Phase 2 — Explainability, Durable Audit & Frontend Consolidation (`P1/P2`)

1. **Dynamic Local Explanations (`TreeSHAP`):** Add a `/explain/ml` helper in `ml-service` using `shap.TreeExplainer` over `XGBoost.joblib` / `RandomForest.joblib` so per-claim attributions vary dynamically with dollar amount, encounter modality, and age, summing exactly to predicted risk margin.
2. **Hash-Chained Persistent Ledger:** Replace `audit_log` list in RAM with append-only SQLite table storing `(id, timestamp, payload_digest, prev_hash, entry_hash, model_key, score, verdict)` linking every record cryptographically (`SHA-256`).
3. **React Component Unification:** Wire `useApiHealth()` (`30,000 ms` poll against `healthRouter`), `ThemeToggle`, `ScoreGauge` (SVG eased arc animation), and `AuditTrail` directly inside `App.tsx` / `AgenticPage.tsx`, retiring orphaned duplicates cleanly.

### Phase 3 — MLOps Governance, Schema Parity & Drift Telemetry (`P3`)

1. **Align UI Schema to Track B `intake` Policy (`28f` ex-`Cluster`):** Deploy `submission/`'s `intake` `RandomForest` artifact (`threshold = 0.416667`, validated `AP = 1.0`) behind `/predict/intake` so users do not supply post-hoc `Cluster` IDs at single-claim triage time.
2. **Continuous Data & Concept Drift Monitoring:** Log feature $z$-score envelopes (especially monitoring `ClaimAmount` boundary migration near `\$7,465.07` and rolling `ClaimDate` distribution shifts so neural architectures don't silently decay across future calendar years).
3. **Automated CI Verification Gate:** Run unified pipeline checks on every pull request:
   * `python ml-service/smoke_test.py`
   * `python submission/scripts/verify.py` (`128` rows + `32` API + `4 × HTTP 422` + `10,800` Chromium JS parity checks)
   * `npx tsc --noEmit` + `npm run build` across `backend/node-api` & `frontend`

---

## 3. Master Cross-Approach Takeaway Summary

Across `4,500` claims, `10` trained model architectures across Track A (`5` Classical ML + `5` Deep Learning), `32` controlled benchmark runs in Track B, an Agentic + `gemini-3.8-flash` UI stack, and `10,800` pure-JS edge parity proofs, three enduring engineering lessons stand verified:

1. **Simple Cost-Sensitive Trees Win Mixed Tabular Claims & Out-of-Time Stability:** Orthogonal splits (`XGBoost` `100.00%` Track A Test PR-AUC, `RandomForest` `1.0000` Track B `intake` AP, `1.0000` `temporal` AP across all trees) dramatically outperform naive autoencoders (`5.23%` PR-AUC) and hold steady across future billing dates where `MLP` (`0.8581`) and `ResidualMLP` (`0.7979`) extrapolate poorly.
2. **Ablations Expose Shortcuts That Benchmarks Hide:** Without Track B's systematic `intake` (`ex-Cluster`), `no_amount`, and `temporal` diagnostics (`submission/`), teams would overestimate pre-intake generalizability (`98.89%` fraud concentration in `Cluster == 1`) and overlook `SVC` Platt discrepancies (`51` vs `39` flagged).
3. **Orchestration & Copilot Ergonomics Must Rest on Empirical Ground Truth:** Agentic traces (`4` stages), SHA-256 digests, XAI attribution visuals, and `gemini-3.8-flash` Q&A elevate SIU analyst usability only when backed by real trained predictors rather than threshold-clamped heuristics (`\$300` vs `\$47,033` clamps). Together cleanly integrated, Approaches 1, 2, and 3 form a cohesive, verifiable reference blueprint.
