# Health Insurance Fraud Detection — Comprehensive Tri-Approach Technical Documentation Suite

> **Document Suite Scope:** Complete multi-volume engineering, mathematical, architectural, and empirical reference covering **Approach 1 (Classical Machine Learning)**, **Approach 2 (Deep Learning & Tabular Representation Learning)**, **Approach 3 (Agentic AI, Explainability, & Conversational Copilot)**, the unified **3-Tier Production Web Platform**, and the **Track B Controlled Benchmark & Browser-Inference Laboratory**.
>
> **Workspace Root:** `c:\Users\RAVIPRAKASH\Downloads\mini-project`
> **Academic Context:** CSE Mini-Project, Indian Institute of Information Technology (IIIT) Dharwad — Supervised by **Prof. Ramesh Athe**
> **Source Fidelity Guarantee:** Every file path, symbol name, hyperparameter, REST route, JSON contract, confusion matrix, latency measurement, byte count, and code defect documented across this eight-volume suite is verified directly against source files and saved runtime artifacts in the repository.

---

## 1. Suite Purpose & Reading Strategy

Modern automated fraud detection in healthcare insurance demands three distinct capabilities that no single modeling family satisfies cleanly on its own:

1. **Deterministic High-Precision Discrimination on Tabular Records (Approach 1 — Classical ML):** Tree ensembles and linear/kernel baselines operate directly on engineered tabular features (`ClaimAmount`, demographics, temporal breakdowns, encounter context, and optional cluster context), delivering sub-millisecond scoring and near-perfect discrimination when fraud signals follow sharp axis-aligned splits or monotonic thresholds.
2. **Distributed Representation Learning & Unsupervised Boundary Modeling (Approach 2 — Deep Learning):** Parametric neural networks (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`, and an unsupervised `Autoencoder`) test whether smooth non-linear projections, residual highway paths, adjacent-column 1D convolutions, self-attention across scalar column tokens, or benign-only reconstruction error can generalize when categorical semantics and temporal drift challenge fixed decision boundaries.
3. **Composable Pipeline Orchestration, Auditability, & Analyst Decision Support (Approach 3 — Agentic AI + XAI):** Production claim triage requires multi-stage workflow orchestration, cryptographically indexed audit trails, per-claim feature attribution explanations, and natural-language analyst copilots rather than opaque probability scalars alone.

Rather than merging three experimental tracks prematurely or overwriting historical benchmarks, the repository preserves each contributor's isolated training workspace alongside an integrated three-tier serving platform and a reproducibility-focused controlled benchmark lab. This documentation suite captures the entire engineering estate in full technical depth.

### Recommended Reading Paths by Audience

| Reader Role | Recommended Starting Order | Primary Volumes |
| :--- | :--- | :--- |
| **Faculty Reviewers & Evaluators** | Index → Executive Architecture → Data Audit → Cross-Approach Synthesis | `README.md`, `01`, `02`, `08` |
| **Classical ML & Feature Engineers** | Data Audit → Approach 1 → Track B Controlled Lab | `02`, `03`, `07` |
| **Deep Learning Researchers** | Data Audit → Approach 2 → Track B Temporal & Ablation Study | `02`, `04`, `07` |
| **Agentic & Full-Stack Engineers** | Executive Architecture → Approach 3 → 3-Tier Platform → Audit Ledger | `01`, `05`, `06`, `08` |
| **DevOps / QA & Release Maintainers** | 3-Tier Platform → Track B Verification → Runbooks & Discrepancy Audit | `06`, `07`, `08` |

---

## 2. Multi-Volume Table of Contents

| Volume | Filename | Core Domain | Key Technical Deliverables Covered |
| :---: | :--- | :--- | :--- |
| **Index** | [`README.md`](./README.md) | Master Navigation & Glossary | Suite topology, ownership matrix, quick-reference metrics, dual-track index, glossary of symbols |
| **Vol. 01** | [`01_EXECUTIVE_ARCHITECTURE_AND_LITERATURE_FOUNDATIONS.md`](./01_EXECUTIVE_ARCHITECTURE_AND_LITERATURE_FOUNDATIONS.md) | System & Research Foundations | Project thesis, tri-approach taxonomy, Dual-Track A vs. Track B governance, 6-pillar literature review (20 citations), 13 ADRs from `DECISIONS.md`, Mermaid topologies |
| **Vol. 02** | [`02_MATHEMATICAL_FORMULATION_DATA_AUDIT_AND_CONTRACTS.md`](./02_MATHEMATICAL_FORMULATION_DATA_AUDIT_AND_CONTRACTS.md) | Theory, Dataset Audit & Schemas | Class-skew ERM formulation, weighted BCE derivation, Autoencoder MSE & percentile thresholding, PR-AUC vs. ROC-AUC proofs, 19-column workbook SHA-256 audit, `Cluster`/`ClaimStatus`/`ClaimAmount` leakage proofs, 29-feature schema & worked transformations |
| **Vol. 03** | [`03_APPROACH_1_CLASSICAL_MACHINE_LEARNING.md`](./03_APPROACH_1_CLASSICAL_MACHINE_LEARNING.md) | Classical ML (`Approch-1/`) | End-to-end `eda_plots.py`, `preprocess.py`, `train_all_models.py`, `comparison_plots.py`, `error_analysis.py`, `create_notebooks.py`; math & hyperparams for `LR`, `RF`, `XGB`, `SVC`, `HistGradientBoosting`; Platt scaling mismatch; Windows `project/` junction runbook |
| **Vol. 04** | [`04_APPROACH_2_DEEP_LEARNING.md`](./04_APPROACH_2_DEEP_LEARNING.md) | Deep Learning (`Approch-2-DL/`) | `dl_common.py`, `submission_models.py`, `dl_advanced_models.py`, `train_mlp.py`, `train_supervised.py`, `train_advanced.py`, `evaluate_mlp.py`; tensor shapes & exact parameter counts (`4,033` to Transformer); `Autoencoder` `29→32→10→32→29` analysis; Track A metrics & failure analysis |
| **Vol. 05** | [`05_APPROACH_3_AGENTIC_AI_XAI_AND_COPILOT.md`](./05_APPROACH_3_AGENTIC_AI_XAI_AND_COPILOT.md) | Agentic AI (`Approach-3-Agentic-AI/` & Copilot) | `:8020` FastAPI prototype vs. Node `predict.ts` shadow engine; 6-role conceptual pipeline vs. 4-event runtime trace; closed-form `\$250` threshold saturation proof; `XAIExplainer` static attribution vs. Shapley axioms; SHA-256 `audit_log`; `gemini-3.8-flash` chat route; mounted vs. unmounted frontend UI audit |
| **Vol. 06** | [`06_FULL_STACK_THREE_TIER_PLATFORM.md`](./06_FULL_STACK_THREE_TIER_PLATFORM.md) | 3-Tier Integration (`ml-service/`, `backend/`, `frontend/`) | FastAPI `:8000` microservice (`claims.py` `extra="forbid"`, `classical_service.py`, `dl_service.py`, `mlp_service.py`), Express `:3000` gateway (`server.ts`, `ml.ts`, `chat.ts`), React `:5173` SPA (`App.tsx`, `ClassicalMLPage.tsx` fallback proof, `DLPage.tsx`, `AgenticPage.tsx`), full REST contract |
| **Vol. 07** | [`07_TRACK_B_CONTROLLED_BENCHMARK_AND_BROWSER_LAB.md`](./07_TRACK_B_CONTROLLED_BENCHMARK_AND_BROWSER_LAB.md) | Controlled Lab (`submission/`) | Scientific benchmark across 4 scenarios (`baseline`, `intake`, `no_amount`, `temporal`) × 8 models = 32 runs (`128` metric rows); deterministic PyTorch vs. Track A; validation-F1 thresholding; `Row 1124` forensics; `1,000`-replicate bootstrap CIs; pure-JS browser export & `10,800`-score parity proof |
| **Vol. 08** | [`08_CROSS_APPROACH_SYNTHESIS_INVENTORY_RUNBOOKS_AND_AUDIT.md`](./08_CROSS_APPROACH_SYNTHESIS_INVENTORY_RUNBOOKS_AND_AUDIT.md) | Synthesis, Inventory, Ops & Audit | Tri-approach comparison matrix, complete directory/file line-count inventory, environment & PowerShell verification runbooks, **Verified Code-vs-Docs Discrepancy Ledger**, production hardening roadmap |

---

## 3. Team Ownership & Contribution Matrix

Project leadership, experimentation, and full-stack delivery are partitioned across three contributors under faculty supervision, establishing clear ownership boundaries between isolated model research folders and shared platform layers:

| Contributor / Advisor | Roll Number / Role | Primary Ownership Domains | Key Directories Owned |
| :--- | :--- | :--- | :--- |
| **J. Ganesh Kumar Reddy** | `23BDS024` (Student Contributor) | Approach 1 Classical ML pipeline, EDA & comparison visualizations, Track B controlled benchmark harness, browser inference exporter, parity verifier | `Approch-1/`, `submission/` |
| **M. Jagadeeswar** | `23BDS033` (Student Contributor) | Approach 2 PyTorch Deep Learning suite (5 neural architectures), 29-feature contract alignment, FastAPI ML inference tier, Node/React integration & ground-truth validation | `Approch-2-DL/`, `ml-service/`, `backend/node-api/` (shared routes), `frontend/` (ML/DL pages & API layer) |
| **B. Varshith** | `23BDS011` (Student Contributor) | Approach 3 Agentic orchestration prototype (`:8020`), XAI attribution & SHA-256 audit logging service, Gemini conversational copilot, Node agentic shadow scorer, agentic React dashboard modules | `Approach-3-Agentic-AI/`, `backend/node-api/src/agentic/`, `backend/node-api/src/routes/chat.ts`, `frontend/src/agentic/`, `frontend/src/pages/AgenticPage.tsx` |
| **Prof. Ramesh Athe** | Faculty Guide (CSE, IIIT Dharwad) | Problem scoping, methodological supervision, leakage/temporal/ablation evaluation criteria, final review | Academic governance across `docs/` & `submission/` |

---

## 4. High-Level Architectural Overview

The repository realizes **two complementary runtime philosophies** built over a single canonical raw workbook (`Health Insurance Fraud Claims.xlsx`, `4,500` claims, `19` columns, `6.00%` fraud prevalence):

```text
+---------------------------------------------------------------------------------------------------+
|                        RAW SOURCE DATASET: Health Insurance Fraud Claims.xlsx                     |
|          4,500 Claims x 19 Columns | 270 Frauds (6.00%) | 4,230 Legitimate | 0 Missing            |
+------------------------------------+-------------------------------------------+------------------+
                                     |                                           |
        [Track A: Exploratory & Integrated 3-Tier Stack]            [Track B: Controlled Ablation Lab]
                                     |                                           |
     +-------------------------------+-------------------------------+           v
     |                               |                               |   submission/scripts/run_pipeline.py
     v                               v                               v   4 Scenarios x 8 Models = 32 Runs
Approch-1/ (5 Classical)    Approch-2-DL/ (5 Neural)     Approach-3-Agentic-AI/  (baseline, intake,
StandardScaler + OHE (29f)  MLP, ResMLP, CNN1D,          FastAPI :8020 Proto +      no_amount, temporal)
LR, RF, XGB, SVC, HGB       TabTransformer, Autoencoder  Node :3000 Agentic Chain  + Browser JS Export :8080
     |                               |                               |   + 10,800-Score Parity Check
     +---------------+---------------+---------------+---------------+
                     |                               |
                     v                               v
        ml-service/ (FastAPI :8000)      backend/node-api/ (Express :3000)
        Loads Classical .joblib + DL .pt  Proxies /api/ml/* -> :8000
                     |                   Serves /api/agentic/predict (local)
                     |                   Serves /api/chat (gemini-3.8-flash)
                     +---------------+---------------+
                                     |
                                     v
                        frontend/ (React 19 + Vite :5173)
                  Tabs: Overview | Classical ML | Deep Learning | Agentic AI
                        (+ ChatBox copilot visible on Agentic AI tab)
```

---

## 5. Executive Cross-Approach Quantitative Snapshot

The summary tables below establish exact headline benchmarks directly verified in the repository logs before detailed analysis in Volumes `02` through `07`.

### 5.1 Approach 1 — Classical Machine Learning (Track A Test Set, `N = 675`, `40` Frauds)

| Model Key (`joblib` Artifact) | Underlying Estimator | Val F1 | Val PR-AUC | Test Acc | Test Prec | Test Rec | Test F1 | Test ROC-AUC | Test PR-AUC | Fit Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LogisticRegression` | `LogisticRegression(max_iter=2000, class_weight='balanced')` | 84.21% | 94.11% | 96.44% | 62.50% | **100.00%** | 76.92% | 99.50% | 91.86% | 0.013 |
| `RandomForest` | `RandomForestClassifier(n_estimators=300, class_weight='balanced')` | 98.77% | **100.00%** | **99.85%** | **100.00%** | 97.50% | 98.73% | 99.996% | 99.94% | 0.625 |
| `XGBoost` | `XGBClassifier(scale_pos_weight=2961/189)` | 98.77% | 99.94% | **99.85%** | 97.56% | **100.00%** | **98.77%** | **100.00%** | **100.00%** | 0.028 |
| `SVM` | `SVC(probability=True, class_weight='balanced', kernel='rbf')` | 87.91% | 94.42% | 97.78% | 74.51% | 95.00% | 83.52% | 99.55% | 92.52% | 0.285 |
| `GradientBoosting` *(Val Best)* | `HistGradientBoostingClassifier(class_weight='balanced')` | **100.00%** | **100.00%** | **99.85%** | **100.00%** | 97.50% | 98.73% | 98.47% | 97.65% | 0.192 |

### 5.2 Approach 2 — Deep Learning (`Approch-2-DL/`, Track A Test Set, `N = 675`, `40` Frauds)

| Architecture (`models/<key>_best.pt`) | Paradigm / Supervision | Stop Epoch | Train Time (s) | Test Acc | Test Prec | Test Rec | Test F1 | Test ROC-AUC | Test PR-AUC | Total Infer Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MLP` | 3-Layer Dense Neural Net (`29→64→32→1`) | 34 / 49 | 4.43 | **98.81%** | **84.78%** | 97.50% | 90.70% | 99.73% | 95.38% | 0.732 ms |
| `ResidualMLP` | Skip-Connected Highway MLP (`2` ResidualBlocks) | 20 / 35 | — | 98.52% | 80.00% | **100.00%** | 88.89% | 99.75% | 95.61% | 0.674 ms |
| `CNN1D` | 1D Convolutions across Ordered Features | 41 / 56 | — | **98.81%** | 83.33% | **100.00%** | **90.91%** | 99.92% | 98.82% | 0.766 ms |
| `TabTransformer` | 2-Layer Feature-Token Multi-Head Attention | 34 / 49 | — | 98.67% | 81.63% | **100.00%** | 89.89% | **99.94%** | **99.11%** | 18.647 ms |
| `Autoencoder` | Unsupervised Benign Reconstruction (`29→32→10→32→29`) | 35 | — | 90.37% | 3.70% | 2.50% | 2.99% | 46.02% | 5.23% | **0.498 ms** |

### 5.3 Approach 3 & Track B Highlights

* **Approach 3 (`AgenticEnsemble-v1` across Node `:3000` & Python `:8020`):** Executes a deterministic 4-stage pipeline (`ingestion` → `validation` → `risk-scorer` → `xai-explainer`) coupled with 12-character SHA-256 `audit_log` digests, static `XAIExplainer` feature attributions, and a `gemini-3.8-flash` analyst copilot (`POST /api/chat`). Unlike Approaches 1 & 2, its active risk formula `min(max((claim_amount * 0.01) + (patient_age * 0.005) - 2.0, 0.0), 1.0)` is an uncalibrated rule stub that saturates to `1.0` once `claim_amount >= 250 - 0.5 * patient_age`.
* **Track B Controlled Lab (`submission/`):** Benchmarks `5` Classical + `3` Deep Learning models across `4` scenarios (`32` runs total). Proves **validation-F1 threshold selection** selects `intake` `RandomForest` (`threshold = 0.4167`, test `AP = 1.0`, `F1 = 0.987342`, 1 FN at `source_row = 1124`), reveals catastrophic collapse without `ClaimAmount` (`AP` drops from `~1.0` to `0.1869–0.2404`), and exposes **temporal generalization gaps** where tree ensembles retain `AP = 1.0, F1 = 1.0` post-`2024-03-11` while neural networks fall (`CNN1D` `0.9172`, `MLP` `0.8581`, `ResidualMLP` `0.7979`).

---

## 6. Dual-Track Repository Architecture (`Track A` vs. `Track B`)

The workspace maintains strict isolation between historical/integrated development (**Track A**) and controlled scientific ablation (**Track B**):

| Dimension | Track A (`Approch-1/`, `Approch-2-DL/`, `ml-service/`, `backend/`, `frontend/`) | Track B (`submission/`) |
| :--- | :--- | :--- |
| **Primary Mandate** | Preserve individual teammate deliverables and power the live 3-tier full-stack dashboard | Produce reproducible ablation, bootstrap, temporal, and zero-dependency browser parity proofs |
| **Directory Policy** | Treated as read-only upstream artifacts by `submission/` | Completely self-contained under `submission/` with its own pipeline & static bundle |
| **Preprocessor Artifacts** | `Approch-1/models/preprocessor.joblib` (`29` features incl. `Cluster`) | Per-scenario `contract.json` (`baseline` 29f, `intake` ex-`Cluster`, `no_amount`, `temporal`) |
| **PyTorch Execution Mode** | Standard CUDA/CPU execution (`dl_common.py`, `seed=42`) | Strict deterministic algorithms (`torch.use_deterministic_algorithms(True)`) |
| **Classification Threshold** | Fixed `0.5` probability (`predict` / `probs >= 0.5`) or 95th-percentile MSE (`Autoencoder`) | Validation-set grid search maximizing F1 per model/scenario (e.g., `RF` `0.4167`) |
| **Serving Runtime** | FastAPI `:8000` → Node Express `:3000` → React `:5173` (+ `:8020` prototype) | Pure-Python HTTP server `:8080` + pure-JS browser inference weights (`export_browser.py`) |

---

## 7. Quick-Reference File & Directory Map

| Path | Role | Details Covered In |
| :--- | :--- | :--- |
| `README.md`, `PROJECT_PROGRESS.md`, `DECISIONS.md` | Root setup, team milestones, 13 architectural decisions | Vol. 01, Vol. 08 |
| `Health Insurance Fraud Claims.xlsx` (copy inside `Approch-1/data/raw/`) | Canonical 4,500-claim workbook (`SHA-256: 952b6e...`) | Vol. 02, Vol. 03 |
| `Approch-1/scripts/*.py`, `models/*.joblib`, `evaluation/*` | Classical ML preprocessing, 5-model training, plots, error analysis | Vol. 03 |
| `Approch-2-DL/scripts/*.py`, `models/*.pt`, `results/*_metrics.json` | PyTorch DL training, contract metadata, Track A evaluations | Vol. 04 |
| `Approach-3-Agentic-AI/app/*.py`, `run_agentic.py` | Standalone FastAPI `:8020` agentic chain + XAI + audit log | Vol. 05 |
| `ml-service/app/*.py`, `smoke_test.py` | Production Python ML/DL inference microservice (`:8000`) | Vol. 06 |
| `backend/node-api/src/**/*.ts` | Express API gateway (`:3000`), ML proxy, Agentic shadow engine, Gemini chat | Vol. 05, Vol. 06 |
| `frontend/src/**/*.tsx` | React 19 + TypeScript + Tailwind SPA (`:5173`) & unmounted modules | Vol. 05, Vol. 06 |
| `submission/scripts/*.py`, `results/*.json`, `results/metrics.csv` | Controlled 32-run benchmark, bootstrap CIs, browser JS parity checks | Vol. 07 |
| `docs/00`–`05`, `docs/ALL_THREE_APPROACHES_DOCUMENTATION.md` | Earlier project docs & incomplete draft audited for factual parity | Vol. 08 |

---

## 8. Glossary of Symbols, Metrics & Abbreviations

| Term / Symbol | Exact Definition in This Repository |
| :--- | :--- |
| **PR-AUC / AP** | Precision-Recall Area Under Curve (`average_precision_score`). Primary discrimination metric because `94%` of claims are legitimate (`prevalence = 0.06`). |
| **ROC-AUC** | Receiver Operating Characteristic Area Under Curve (`roc_auc_score`). Sensitive to `TN` abundance under heavy class skew. |
| **`pos_weight`** | BCE loss minority-class multiplier computed on training set as $N_{\text{neg}} / N_{\text{pos}} = 2961 / 189 \approx 15.6667$. |
| **`extra="forbid"`** | Pydantic schema configuration in `ml-service/app/schemas/claims.py` rejecting any unknown JSON body property with HTTP `422`. |
| **Track A / Track B** | Dual-track split: Track A = original approach workspaces + live 3-tier app; Track B = isolated controlled lab in `submission/`. |
| **Intake Policy** | Deployment feature policy excluding `Cluster` (post-hoc unsupervised grouping unavailable at live single-claim intake time). |
| **Temporal Split** | Chronological holdout partitioned at dates `2023-11-21` (val start) and `2024-03-10` (val end), testing out-of-time robustness (`N_test = 671`). |
| **Platt Scaling** | Logistic calibration curve fitted internally by `SVC(probability=True)`, explaining mismatches between `SVC.predict()` (`51` flagged) and `SVC.predict_proba() >= 0.5` (`39` flagged). |
| **XAI / Audit Digest** | Per-claim feature contribution vector returned by `XAIExplainer` / `predict.ts` alongside a 12-character truncated SHA-256 hex digest. |
