# Volume 08 — Cross-Approach Synthesis, Full Code Inventory, Runbooks & Verified Discrepancy Audit

> **Document Classification:** Executive Decision Matrix, Complete Asset Catalog & Truth-Audit Ledger  
> **Scope:** Synthesizes **Approach 1 (Classical ML)**, **Approach 2 (Deep Learning)**, **Approach 3 (Agentic AI + Copilot)**, **Track A Three-Tier Serving**, and **Track B Controlled Lab**, concluding with a source-verified discrepancy audit resolving every misstatement across earlier drafts.  
> **Primary Ground-Truth Sources:** Direct repository source verification across `README.md`, `PROJECT_PROGRESS.md`, `DECISIONS.md`, `docs/*`, `Approch-1/*`, `Approch-2-DL/*`, `Approach-3-Agentic-AI/*`, `ml-service/*`, `backend/node-api/*`, `frontend/*`, and `submission/*`.

---

## 1. Tri-Approach Engineering Decision Matrix

When selecting an architecture for production healthcare claims adjudication, no single paradigm dominates every dimension simultaneously. The comparison table below synthesizes empirical and systems facts established across Volumes `01` through `07`:

| Engineering Dimension | Approach 1: Classical ML (`Approch-1/`) | Approach 2: Deep Learning (`Approch-2-DL/`) | Approach 3: Agentic + XAI (`Approach-3` / `node-api`) | Recommended Production Synthesis |
| :--- | :--- | :--- | :--- | :--- |
| **Peak Test PR-AUC (Track A)** | **`100.00%`** (`XGBoost`) / `99.94%` (`RF`) | `99.11%` (`TabTransformer`) / `98.82%` (`CNN1D`) | Heuristic stub only (`\$250–\$300` cap) | Classical Tree Scorer (`XGB` / Track B `intake RF`) |
| **Peak Test F1 (`N=675`)** | **`98.77%`** (`XGB`) / `98.73%` (`RF`, `HGB`) | `90.91%` (`CNN1D`) / `90.70%` (`MLP`) | Inapplicable (uncalibrated stub) | Classical Tree + Validation Threshold (`ADR-12`) |
| **Temporal Out-of-Time Stability (`>= 2024-03-11`)** | **`1.0000` AP / `1.0000` F1** (`RF`, `XGB`, `HGB`) | `0.7979` (`ResMLP`) – `0.9172` (`CNN1D`) AP | Static time-invariant stub | Orthogonal Trees resist `ClaimYear` drift |
| **Pre-Clustering Intake Viability (`ex-Cluster`)** | **`1.0000` Test AP / `0.9873` F1** (`intake RF`) | Higher threshold variance without trees | Drops `Cluster` in scorer (uses only `amt, age`) | Mandatory `intake` policy (`ADR-07`) |
| **Unsupervised Anomaly Detection** | Not natively unsupervised | **`5.23%` PR-AUC, `MCC=-0.0192`** (`Autoencoder` fail) | Not implemented | Avoid global L2 Autoencoder when frauds overlap normal dollar range |
| **Probability Calibration Behavior** | Best rank separation; `SVC` Platt-vs-margin caveat | Shifted log-odds (`+2.7515`) from `pos_weight=15.67` | Clamped heuristic `[0, 1]` | Apply validation F1 tuning + isotonic/Platt calibration audit (`ADR-05`) |
| **Inference Latency Footprint** | **`0.0003s–0.0537s` / `100` claims** (`3 µs–537 µs`) | `0.498 ms–18.647 ms` batch (`TabTrans` `24×` slower) | Instant JS arithmetic (`predict.ts`) | FastAPI Tree inference (`< 1 ms`) + async trace overhead |
| **Model Storage Footprint** | `1.1 KB` (`LR`) to `3.31 MB` (`RF`) (`HGB` `70.7 KB`) | `14.1 KB` (`AE`) to `343.7 KB` (`TabTrans`) | Code-only (`< 2 KB`) | `HGB` (`70.7 KB`) / `XGB` (`83.8 KB`) for slim containers; `RF` pure-JS edge |
| **Feature Attribution & Analyst UX** | Feature importance only | Opaque tensors without wrappers | **`4`-stage trace, SHA-256 digest, `XAIExplainer`, `gemini-3.8-flash` ChatBox** | Wrap Classical Tree + `TreeSHAP` inside Approach 3 UX & trace ledger |

---

## 2. Recommended Unified Hybrid Production Blueprint

Bridging the complementary strengths of all three approaches produces an ideal enterprise claim intelligence architecture requiring zero new research discoveries — merely connecting components already built inside this repository:

```text
+---------------------------------------------------------------------------------------------------+
|                   RECOMMENDED UNIFIED ENTERPRISE TRI-APPROACH TARGET ARCHITECTURE                 |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [1] React Presentation Tier (:5173)                                                              |
|      Unified Claim Screening View + Trace Stepper + Dynamic SHAP Chart + ChatBox (gemini-3.8)     |
|                                               |                                                   |
|                                               v                                                   |
|  [2] Express API Gateway (:3000)                                                                  |
|      - Strict Zod Request Defense matching Pydantic `extra="forbid"`                              |
|      - `predict.ts` orchestrates 4 Pipeline Stages via HTTP/RPC instead of inline heuristic:      |
|          Stage 1 (ingestion)    -> Validate Claim Intake Schema (28f `intake` policy ex-Cluster)  |
|          Stage 2 (validation)   -> Guard Date/Numeric Ranges & Category Sets                      |
|          Stage 3 (risk-scorer)  -> POST :8000/predict/ml (XGBoost or Track B `intake RandomForest`|
|                                    at Validation Threshold 0.4167 + optional `CNN1D` 2nd opinion) |
|          Stage 4 (xai-explainer)-> Compute real `TreeSHAP` vector per claim (summing to f(x)-E[f])|
|      - Persist full 64-char SHA-256 Hash-Chained Audit Log to SQLite / WAL (not RAM list)         |
|      - Pass real TreeSHAP top drivers + model score as grounded context into `gemini-3.8-flash`   |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Complete Repository File & Symbol Line-Count Inventory

Below is the audited master file inventory across every active source directory (`Approch-1/`, `Approch-2-DL/`, `Approach-3-Agentic-AI/`, `ml-service/`, `backend/node-api/`, `frontend/`, `submission/`, `docs/`), using line counts verified from direct file inspection:

### 3.1 Root & Project Documentation (`./` & `./docs/`)

| File Path | Verified Lines | Primary Purpose |
| :--- | :---: | :--- |
| `README.md` | `184` | Multi-terminal platform setup guide, architecture summary, PowerShell smoke workflow |
| `PROJECT_PROGRESS.md` | `185` | Team ownership, integration timeline, Track A metrics, spot-check notes (`Index 0`, `Index 2`) |
| `DECISIONS.md` | `15` | 13 core architectural decisions (`ADR-01` through `ADR-13`) |
| `docs/README.md` | `32` | Deliverables documentation directory index |
| `docs/01_LITERATURE_SURVEY.md` | `75` | 6-pillar literature review spanning `20` references |
| `docs/02_PROBLEM_FORMULATION.md` | `82` | Loss & metric mathematical formulation overview |
| `docs/03_INNOVATION_AND_IMPLEMENTATION.md` | `71` | Implementation & Track B summary (audited below for hyperparameter errata) |
| `docs/04_FULL_PROJECT_DOCUMENTATION.md` | `152` | Earlier consolidated overview (audited below) |
| `docs/05_SYSTEM_ARCHITECTURE_FRONTEND_BACKEND_GUIDE.md` | `123` | Earlier full-stack UI guide (audited below for unmounted/non-existent symbols) |
| `docs/ALL_THREE_APPROACHES_DOCUMENTATION.md` | `1,780` | Earlier incomplete single-file draft truncated mid-Part VI inside Agentic AI |

### 3.2 Approach 1 (`Approch-1/`)

| File Path | Verified Lines | Purpose |
| :--- | :---: | :--- |
| `Approch-1/README.md` | `63` | Subsystem runbook & artifact index |
| `Approch-1/requirements.txt` | `26` | Python dependencies (`scikit-learn`, `xgboost`, plotting stack) |
| `Approch-1/data/processed/feature_names.txt` | `29` | Canonical `29` feature order specification |
| `Approch-1/scripts/eda_plots.py` | `109` | Exploratory plots (distributions, imbalance, correlations) |
| `Approch-1/scripts/preprocess.py` | `98` | Split (`70/15/15`), calendar parsing, leaky drops, `ColumnTransformer` |
| `Approch-1/scripts/train_all_models.py` | `148` | Trains `5` estimators, benchmarks timing/size, writes JSON logs |
| `Approch-1/scripts/comparison_plots.py` | `72` | Multi-model ROC, PR, F1, confusion charts |
| `Approch-1/scripts/error_analysis.py` | `55` | Inspects selected `GradientBoosting` test misclassifications |
| `Approch-1/scripts/create_notebooks.py` | `32` | Programmatic `.ipynb` script (**contains unformatted `{class_def}` bug**) |
| `Approch-1/evaluation/model_results.json` | `77` | Complete Validation & Test JSON benchmark records |
| `Approch-1/evaluation/result_analysis.md`, `.txt` | `21`, `13` | Concise error & ranking summaries |
| `Approch-1/logs/training_metrics.json` | `125` | Runtime environment (`Python 3.11.2`, `sklearn 1.9.1`), wall clock `4.12s`, bytes & metrics |
| `Approch-1/documentation/{02,04,05,06,07}_*.md` | `89,65,69,133,51` | Approach 1 dataset, methodology, results (`723,352` typo), Windows junction guide, notes |

### 3.3 Approach 2 (`Approch-2-DL/`)

| File Path | Verified Lines | Purpose |
| :--- | :---: | :--- |
| `Approch-2-DL/README.md` | `139` | DL setup, architecture & evaluation documentation |
| `Approch-2-DL/requirements.txt` | `7` | PyTorch DL requirements |
| `Approch-2-DL/processed/contract_metadata.json` | `49` | Upstream `.npy` feature contract & class counts |
| `Approch-2-DL/scripts/dl_common.py` | `170` | DataLoaders, `MLP` (`4,033` params), seed `42`, unified metrics/timings |
| `Approch-2-DL/scripts/submission_models.py` | `30` | `ResidualMLP` (`18,625` params) & `CNN1D` (`15,345` params) |
| `Approch-2-DL/scripts/dl_advanced_models.py` | `116` | `TabTransformer` (`d_model=32, 2 layers`) & `Autoencoder` (`29→32→10→32→29`) |
| `Approch-2-DL/scripts/train_mlp.py` | `153` | Standalone MLP training script (`logs/training_history.json`) |
| `Approch-2-DL/scripts/train_supervised.py` | `173` | `ResidualMLP` & `CNN1D` shared training runner |
| `Approch-2-DL/scripts/train_advanced.py` | `302` | `TabTransformer` + benign-only `Autoencoder` training & 95th percentile cutoff |
| `Approch-2-DL/scripts/evaluate_mlp.py` | `146` | Post-training evaluation & smoke tester |
| `Approch-2-DL/results/{mlp,residualmlp,cnn1d,tabtransformer,autoencoder}_metrics.json` | `71,65,65,65,68` | Track A DL evaluation JSONs + confusion matrices + latencies |
| `Approch-2-DL/results/inference_smoke_test.json` | `9` | Batch inference smoke check artifact |
| `Approch-2-DL/logs/mlp_training_summary.json`, `training_history.json` | `17`, `394` | MLP checkpoint convergence log & full `49`-epoch history |

### 3.4 Approach 3 & Three-Tier Web Platform (`Approach-3-Agentic-AI/`, `ml-service/`, `backend/`, `frontend/`)

| File Path | Verified Lines | Purpose |
| :--- | :---: | :--- |
| `Approach-3-Agentic-AI/app/main.py` | `24` | Standalone `:8020` FastAPI prototype |
| `Approach-3-Agentic-AI/app/api/predict.py` | `50` | Heuristic pipeline + `XAIExplainer` + `audit_log` (**duplicate `router`, invalid `BackgroundTask`**) |
| `Approach-3-Agentic-AI/app/services/xai_service.py` | `43` | Static attribution weights & `12`-char SHA-256 `audit_log` |
| `Approach-3-Agentic-AI/run_agentic.py`, `requirements.txt`, `docs/README.md` | `12`, `4`, `25` | Launcher (**broken `include_router(predict_fraud)`**), deps, prototype doc |
| `ml-service/README.md`, `requirements.txt`, `smoke_test.py` | `36`, `7`, `46` | Serving docs, deps, integration smoke verifier |
| `ml-service/app/{main.py, schemas/claims.py, api/predictions.py}` | `8`, `56`, `62` | `:8000` FastAPI bootstrap, strict `extra="forbid"`, `/predict{/ml,/dl}` routes |
| `ml-service/app/services/{classical_service,dl_service,mlp_service}.py` | `117`, `187`, `146` | Estimator loading, 29f transform, PyTorch eval + Autoencoder sigmoid scaling |
| `backend/node-api/{package.json, README.md}` | `24`, `33` | Express `:3000` gateway manifest & guide |
| `backend/node-api/src/{server.ts, shared/schemas.ts, routes/ml.ts, routes/chat.ts, agentic/predict.ts}` | `21`, `25`, `111`, `47`, `39` | Server init, TS contracts, FastAPI proxy, `gemini-3.8-flash` chat, in-process Agentic scorer |
| `frontend/{package.json, index.html, vite.config.ts}` | `22`, `15`, `19` | React `19` + Vite `:5173` build & dev `/api` proxy config |
| `frontend/src/{App.tsx, types.ts, lib/api.ts}` | `63`, `82`, `172` | Active tab router, TS response/claim contracts, typed API client |
| `frontend/src/pages/{HomePage, ClassicalMLPage, DLPage, AgenticPage}.tsx` | `167`, `405`, `349`, `247` | All `4` mounted pages (`ClassicalMLPage` includes `0.01`-clamped offline fallback) |
| `frontend/src/agentic/{data/agents.ts, components/ChatBox.tsx, components/AuditTrail.tsx, pages/AgenticDashboard.tsx, pages/ClaimInputForm.tsx}` | `56`, `309`, `157`, `142`, `260` | `ChatBox` mounted on `agentic`; `AgenticDashboard`, `ClaimInputForm`, `AuditTrail` unmounted |

### 3.5 Track B Controlled Benchmark (`submission/`)

| File Path | Verified Lines | Purpose |
| :--- | :---: | :--- |
| `submission/README.md`, `START_DEMO.ps1` | `100`, `9` | Controlled benchmark doc & `:8080` single-command launcher |
| `submission/scripts/{common, run_pipeline, export_browser, verify, serve, make_figures, evaluate_legacy}.py`, `check_browser.mjs` | `63,161,62,48,62,58,31`, `21` | Full deterministic `32`-run trainer, JS exporter, figure renderer, parity/HTTP verifier |
| `submission/models/intake/contract.json` | `59` | Ex-`Cluster` deployment schema contract |
| `submission/results/{dataset_audit, analysis, bootstrap, browser_parity, selection, verification, run_manifest, legacy_artifact_audit, deployment}.json` | `134,147,20,128,8,168,32,116,9` | Complete auditable JSON outputs (`952b6e...`, `Row 1124`, `B=1000`, `10,800` browser scores, etc.) |
| `submission/results/metrics.csv`, `baseline_*_training.json` | `129`, `~301–394` | Complete `128`-row CSV metrics matrix & deterministic PyTorch histories |

---

## 4. Verified Code-vs-Documentation Discrepancy Ledger

To establish a single authoritative ground truth across the repository, the ledger below records **every verified contradiction** between prior documentation files (`docs/01`–`05`, `Approch-1/documentation/05_results.md`, stale summaries) and actual executable code or runtime JSON logs:

| # | Historical / Stale Claim | Actual Ground Truth Observed in Source Code & Logs | Exact Authoritative Proof File |
| :---: | :--- | :--- | :--- |
| **`DISC-01`** | "`Autoencoder` compresses `29 → 16 → 8 → 16 → 29`" (`docs/02`, `docs/03`) | Actual PyTorch topology is **`29 → 32 → 10 → 32 → 29`** (`Linear(29,32)->ReLU->Linear(32,10)` + symmetric decoder). | [`Approch-2-DL/scripts/dl_advanced_models.py`](../../Approch-2-DL/scripts/dl_advanced_models.py) L60–116 |
| **`DISC-02`** | "`MLP` includes `BatchNorm1d` layers" (`docs/03`) | `MLP` uses **`Linear(29,64)->ReLU->Dropout(0.2)->Linear(64,32)->ReLU->Dropout(0.2)->Linear(32,1)`** (`4,033` params) with **zero `BatchNorm1d` layers**. | [`Approch-2-DL/scripts/dl_common.py`](../../Approch-2-DL/scripts/dl_common.py) L50–64 |
| **`DISC-03`** | "Gemini chat uses `gemini-2.0-flash`" (`docs/01`, `03`, `04`, `05`) | Express chat handler configures model **`"gemini-3.8-flash"`** and reads `GEMINI_API_KEY` from `../../Approach-3-Agentic-AI/.env`. | [`backend/node-api/src/routes/chat.ts`](../../backend/node-api/src/routes/chat.ts) L1–47 |
| **`DISC-04`** | "Frontend renders `AgenticDashboard`, `benchmarks.ts`, `ClaimFieldsForm.tsx`, `PredictionResultCard.tsx`" (`docs/05`) | `App.tsx` mounts **`HomePage`, `ClassicalMLPage`, `DLPage`, `AgenticPage`**, plus **`ChatBox`**. `AgenticDashboard`, `ClaimInputForm`, `AuditTrail`, `AgentCard`, `ScoreGauge` exist on disk **unmounted**; `benchmarks.ts` / `ClaimFieldsForm` / `PredictionResultCard` do not exist. | [`frontend/src/App.tsx`](../../frontend/src/App.tsx) L1–63 & file tree |
| **`DISC-05`** | "Track B uses `RF(200)`, `XGB(lr=0.05, 200)`, `HGB(200)`" (`docs/03`) | `submission/scripts/run_pipeline.py` trains **`RF(n_estimators=300)`**, **`XGB(n_estimators=100, max_depth=6, learning_rate=0.3)`**, **`HGB(max_iter=100)`**, `SVC(probability=False)+expit`. | [`submission/scripts/run_pipeline.py`](../../submission/scripts/run_pipeline.py) |
| **`DISC-06`** | "`GradientBoosting.joblib` occupies `723,352` bytes" (`05_results.md`) | Verified serialized byte size is **`72,352` bytes** (`10×` typographic inflation in prior writeup). | [`Approch-1/logs/training_metrics.json`](../../Approch-1/logs/training_metrics.json) L90 |
| **`DISC-07`** | "`Approach-3-Agentic-AI` works via `python run_agentic.py` and `/api/v1/predict` runs `XAIExplainer` seamlessly via Node" | `run_agentic.py` fails (`include_router(predict_fraud)` bug); `predict.py` duplicates `router` (`L6, L30`) & imports invalid `BackgroundTask`; Node `/api/agentic/predict` runs in-process `predict.ts` directly without calling `:8020`, saturating at `1.0` above `\$260–\$300`. | [`Approach-3-Agentic-AI/run_agentic.py`](../../Approach-3-Agentic-AI/run_agentic.py), [`predict.py`](../../Approach-3-Agentic-AI/app/api/predict.py), [`predict.ts`](../../backend/node-api/src/agentic/predict.ts) |

---

## 5. Master Production Hardening & Academic Presentation Checklist

| Priority | Action Item | Files Involved | Verification Criterion |
| :---: | :--- | :--- | :--- |
| **`P0`** | Wire Tier 2 Agentic `risk-scorer` to Tier 1 FastAPI (`POST /predict/ml`, `RandomForest` `intake` / `XGBoost`) | `backend/node-api/src/agentic/predict.ts`, `Approach-3-Agentic-AI/app/api/predict.py` | Authentic fraud/legit probability scores instead of `\$300` rule clamp |
| **`P0`** | Fix `:8020` `run_agentic.py` (`predict.router`) & duplicate `router` + `BackgroundTasks` import in `predict.py` | `Approach-3-Agentic-AI/run_agentic.py`, `app/api/predict.py` | Clean `python run_agentic.py` boot & zero orphaned FastAPI routes |
| **`P1`** | Replace `XAIExplainer` static dict with true `shap.TreeExplainer` per-claim local contributions | `Approach-3-Agentic-AI/app/services/xai_service.py` | $\sum \phi_j$ satisfies Shapley Local Accuracy against predicted margin |
| **`P1`** | Replace `audit_log` RAM list with persistent hash-chained SQLite / WAL table | `Approach-3-Agentic-AI/app/services/xai_service.py`, `node-api` | Survives process restarts; verifiable `prev_hash` audit chain |
| **`P2`** | Align `ClassicalMLPage.tsx` offline demo formula away from `0.01` workbook saturation (`\$47k+`) | `frontend/src/pages/ClassicalMLPage.tsx` L92–98 | Offline demo distinguishes `\$500` vs `\$8,800` claims transparently |
| **`P2`** | Isolate `GEMINI_API_KEY` inside `backend/node-api/.env` + enforce Zod schema validation at `:3000` | `backend/node-api/src/routes/{chat,ml}.ts` | Decouples Express gateway from sibling prototype directories |
| **`P3`** | Surface clear "Intake Policy (`ex-Cluster`)" toggle in React UI matching Track B `selection.json` (`0.4167`) | `frontend/`, `ml-service/` | Aligns live web UI predictions with audited pre-clustering benchmarks |
