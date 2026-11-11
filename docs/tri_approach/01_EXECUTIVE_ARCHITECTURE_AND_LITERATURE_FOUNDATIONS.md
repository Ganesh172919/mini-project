# Volume 01 — Executive Architecture, Research Foundations & Dual-Track Governance

> **Document Classification:** Master Architectural Blueprint & Methodological Foundation  
> **Workspace Scope:** Cross-Repository Governance (`Approch-1/`, `Approch-2-DL/`, `Approach-3-Agentic-AI/`, `ml-service/`, `backend/node-api/`, `frontend/`, `submission/`)  
> **Supervision & Authorship Context:** IIIT Dharwad CSE Mini-Project under **Prof. Ramesh Athe**; executed by **J. Ganesh Kumar Reddy (`23BDS024`)**, **M. Jagadeeswar (`23BDS033`)**, and **B. Varshith (`23BDS011`)**.  
> **Primary Ground-Truth Sources:** [`README.md`](../../README.md), [`PROJECT_PROGRESS.md`](../../PROJECT_PROGRESS.md), [`DECISIONS.md`](../../DECISIONS.md), [`docs/README.md`](../README.md), [`docs/01_LITERATURE_SURVEY.md`](../01_LITERATURE_SURVEY.md), [`docs/02_PROBLEM_FORMULATION.md`](../02_PROBLEM_FORMULATION.md), [`docs/03_INNOVATION_AND_IMPLEMENTATION.md`](../03_INNOVATION_AND_IMPLEMENTATION.md), [`docs/04_FULL_PROJECT_DOCUMENTATION.md`](../04_FULL_PROJECT_DOCUMENTATION.md), and [`docs/05_SYSTEM_ARCHITECTURE_FRONTEND_BACKEND_GUIDE.md`](../05_SYSTEM_ARCHITECTURE_FRONTEND_BACKEND_GUIDE.md).

---

## 1. Executive Mission, Healthcare Fraud Context & Problem Framing

Automated fraud screening in health insurance claims processing requires balancing statistical sensitivity against operational investigation capacity, explainability mandates, and adversarial or chronological shifts in billing behavior. Unlike uniform image recognition or balanced consumer recommendation tasks, structured tabular claims pipelines operate under five non-negotiable engineering realities evidenced directly in this codebase and its canonical dataset (`Health Insurance Fraud Claims.xlsx`):

### 1.1 Severe Minority Class Skew (`6.00%` Prevalence)

Across the `4,500` records stored in `Health Insurance Fraud Claims.xlsx`, exactly `270` claims carry a positive fraud label while `4,230` claims represent legitimate reimbursements (`6.00%` positive class ratio, exact imbalance ratio `15.6667:1` in the `3,150`-row training split where `N_neg = 2,961` and `N_pos = 189`). Under this class prior:

* **Accuracy Degeneracy:** A trivial zero-rule classifier predicting "Legitimate" for every claim attains `94.00%` overall accuracy on test (`635 / 675`) while failing its sole operational purpose with `0.00%` fraud recall.
* **ROC-AUC Optimism:** Because `635` true negatives inflate the false-positive rate denominator ($FPR = FP / (FP + TN)$), moderate absolute false-positive surges barely perturb the Receiver Operating Characteristic area. As observed empirically across our tests, `LogisticRegression` produces `24` false positives alongside `40` true positives (`Precision = 62.50%`), yet still registers `99.50%` Test ROC-AUC compared to `91.86%` PR-AUC.
* **Cost-Sensitive Learning Requirement:** Both Track A and Track B explicitly reweight gradients and impurity splits (`class_weight='balanced'`, `scale_pos_weight = 2961 / 189`, and PyTorch `BCEWithLogitsLoss(pos_weight=2961/189)`) and elevate Precision-Recall Area Under the Curve (`PR-AUC` / `average_precision_score`) as the primary evaluation benchmark.

### 1.2 Mixed Continuous, Cyclical Calendar, Segment & Categorical Modalities

Each raw insurance claim bundles `19` heterogeneous fields spanning distinct statistical scales and semantic roles:

* **Monetary Scale Attributes:** `ClaimAmount` (`\$100.12` to `\$9,997.20`) and `PatientIncome` (`4,500` unique continuous incomes).
* **Demographic & Temporal Coordinates:** `PatientAge` (`100` distinct discrete ages) and `ClaimDate` spanning `731` distinct daily timestamps from `2022-07-09` through `2024-07-08`, engineered into numerical fields `ClaimYear`, `ClaimMonth`, and `ClaimDayOfWeek`.
* **Administrative High-Cardinality Keys:** `ClaimID` (`4,500` unique), `PatientID` (`4,500` unique), `ProviderID` (`4,500` unique), `DiagnosisCode` (`4,495` unique), `ProcedureCode` (`4,495` unique), and `ProviderLocation` (`3,876` unique). Because nearly every row owns distinct identifiers or diagnostic combinations, naive encoding introduces memorization risk and high-dimensional sparsity; both pipelines systematically drop these six identifier/code columns before vectorization.
* **Low-Cardinality Categorical Indicators:** `PatientGender` (`2`), `ProviderSpecialty` (`5`), `PatientMaritalStatus` (`4`), `PatientEmploymentStatus` (`4`), `ClaimType` (`4`), and `ClaimSubmissionMethod` (`3`), expanded via one-hot encoding into `22` binary dimensions.

### 1.3 Post-Hoc Leakage & Single-Claim Intake Mismatch

Raw historical spreadsheets exported after adjudication cycles routinely harbor attributes unavailable—or non-causal—at live pre-payment triage time:

1. **Outcome Leakage (`ClaimStatus`):** `ClaimStatus` (`3` unique values) reflects downstream claim processing state and is purged globally as `LEAKY_COLUMNS = ['ClaimStatus']`.
2. **Batch Clustering Shortcut (`Cluster`):** `Health Insurance Fraud Claims.xlsx` contains a `4`-level integer attribute `Cluster` (`0, 1, 2, 3`). Dataset forensics (`dataset_audit.json`) prove that `267` out of `270` positive frauds (`98.89%`) fall inside `Cluster == 1`, whereas `Cluster == 0` and `Cluster == 2` contain `0` frauds (`1,144` and `1,104` legitimate claims respectively) and `Cluster == 3` contains only `3` frauds (`1,149` legitimate). Relying blindly on `Cluster` creates an illusion of effortless separability during retrospective random-split testing, yet breaks down when evaluating unseen individual claims arriving prior to dataset-level clustering.
3. **Single-Feature Monetary Dominance (`ClaimAmount`):** Fraudulent claims exhibit a compressed high-value window (`min = \$7,465.07`, `max = \$9,978.43`, `mean = \$8,733.57`) inside a broader legitimate envelope (`min = \$100.12`, `max = \$9,997.20`, `mean = \$4,776.80`). Removing `ClaimAmount` altogether collapses every classifier's Average Precision (`0.1869–0.2404`), proving that virtually all predictive discriminability in this dataset derives from structured non-linear interactions keyed to high claim amounts rather than demographics alone.

### 1.4 Temporal Shift & Paradigm Inductive Bias

Stratified random splits (`70% / 15% / 15%`) test interpolation inside a stationary joint distribution. Operational adjudication, however, tests extrapolation forward through calendar time (`2024-03-11` onward in Track B's `temporal` split). Comparative evaluation reveals a sharp divergence: orthogonal tree ensembles preserve `1.0000` AP and `1.0000` F1 across chronological boundaries, whereas smooth continuous neural architectures (`MLP`, `ResidualMLP`, `CNN1D`) experience measurable precision and AP degradation.

### 1.5 Decision Explainability, Provenance & Investigator Ergonomics

High-stakes healthcare claim rejections demand verifiable rationale, immutable processing logs, and interactive workflows for Special Investigations Unit (SIU) analysts. This drives the third pillar of the project: structured workflow orchestration, per-claim risk attribution dictionaries (`XAIExplainer`), cryptographic SHA-256 execution digests (`audit_log`), and a Gemini-backed conversational analyst assistant (`gemini-3.8-flash`).

---

## 2. Tri-Approach Architecture Taxonomy

Rather than advocating one monolithic algorithm, the repository isolates and evaluates **three fundamentally distinct AI engineering paradigms** side-by-side over the identical canonical data problem.

```text
+=======================================================================================================+
|                        CANONICAL DATASET: Health Insurance Fraud Claims.xlsx                          |
|                    4,500 Records | 19 Columns | 270 Frauds (6.00%) | 4,230 Legit                      |
+=======================================================================================================+
           |                                        |                                        |
           v                                        v                                        v
+-----------------------------+      +-----------------------------+      +-----------------------------+
|  APPROACH 1: CLASSICAL ML   |      |  APPROACH 2: DEEP LEARNING  |      |  APPROACH 3: AGENTIC + XAI  |
|  Directory: Approch-1/      |      |  Directory: Approch-2-DL/   |      |  Dir: Approach-3-Agentic-AI |
|  Lead: J. Ganesh Kumar      |      |  Lead: M. Jagadeeswar       |      |  Lead: B. Varshith          |
+-----------------------------+      +-----------------------------+      +-----------------------------+
| Paradigms:                  |      | Paradigms:                  |      | Paradigms:                  |
|  - Regularized Linear (LR)  |      |  - Feed-Forward Dense (MLP) |      |  - 4-Stage Pipeline Chain   |
|  - Bagged Trees (RF)        |      |  - Residual Highway (ResMLP)|      |  - Ingestion & Validation   |
|  - Gradient Boost (XGB,HGB) |      |  - Feature-Window CNN1D     |      |  - Risk Heuristic Scorer    |
|  - Kernel Margin (RBF SVC)  |      |  - Column Attention Transf. |      |  - Static XAI Weight Map    |
|                             |      |  - Benign-Only Autoencoder  |      |  - SHA-256 Audit Ledger     |
|                             |      |                             |      |  - Gemini Copilot (Node)    |
+-----------------------------+      +-----------------------------+      +-----------------------------+
| Track A Key Outcomes:       |      | Track A Key Outcomes:       |      | Track A Key Outcomes:       |
|  - XGB PR-AUC: 100.00%      |      |  - TabTrans PR-AUC: 99.11%  |      |  - Active rule stub satur.  |
|  - HGB Val F1: 100.00%      |      |  - CNN1D F1: 90.91%         |      |    at >= $250 claim_amount  |
|  - RF Test Prec: 100.00%    |      |  - MLP F1: 90.70%           |      |  - Node shadow engine matches|
|  - SVC Platt discrepancy    |      |  - AE PR-AUC: 5.23% (fail)  |      |    Python :8020 formula     |
+-----------------------------+      +-----------------------------+      +-----------------------------+
```

### 2.1 Detailed Paradigm Breakdown: Approach 1 (`Approch-1/`)

Approach 1 applies deterministic tabular feature engineering (`StandardScaler` + `OneHotEncoder`) followed by five classical statistical and ensemble estimators:

1. **`LogisticRegression` (`Linear Baseline`):** Fits an L2-regularized sigmoid hyperplane using L-BFGS (`max_iter=2000`, `class_weight='balanced'`). Serves as the additive log-odds linear reference. Establishes `100.00%` Recall with `62.50%` Precision (`24` false positives), proving that while frauds occupy the upper-tail half-space linearly, legitimate claims in the overlapping upper monetary tail induce significant false alarms unless non-linear gating is introduced.
2. **`RandomForestClassifier` (`Bagged Orthogonal Decision Trees`):** Aggregates `300` decorrelated CART trees trained with balanced class weights. Achieves `100.00%` Test Precision (`0` false positives across `635` legitimate test records), `97.50%` Recall (`1` missed fraud), `99.996%` ROC-AUC, and `99.94%` PR-AUC, while producing the largest disk footprint (`3,465,961 B`) and highest classical batch prediction latency (`0.0537 s` per 100 claims).
3. **`XGBClassifier` (`Regularized Gradient Boosted Trees`):** Sequentially fits shallow boosting trees optimizing balanced binary log-loss with minority ratio `scale_pos_weight = 2961 / 189 ≈ 15.6667`. Delivers peak Track A Test performance: `100.00%` Recall, `97.56%` Precision, `98.77%` F1, `100.00%` ROC-AUC, and `100.00%` PR-AUC in only `0.028 s` training time (`85,788 B`).
4. **`SVC` (`RBF Kernel Maximum-Margin Classifier`):** Maps `29`-dimensional normalized vectors into an infinite-dimensional Gaussian kernel space (`kernel='rbf'`, `probability=True`, `class_weight='balanced'`). Exposes a critical architectural subtlety verified in both Track A and Track B audits: raw margin classification (`SVC.predict`) flags `51` test cases (`Precision 74.51%`, `Recall 95.00%`, `F1 83.52%`), whereas internal Platt-scaled logistic regression (`SVC.predict_proba >= 0.5`) flags `39` test cases (`TN=632, FP=3, FN=4, TP=36`, `F1 91.14%`).
5. **`HistGradientBoostingClassifier` (`Histogram-Binned Boosting`, serialized as `GradientBoosting.joblib`):** Bins continuous inputs into integer histograms prior to gradient boosting (`class_weight='balanced'`). Attained perfect `100.00%` Validation F1 (`Val PR-AUC 100.00%`), triggering automated selection as `best_model.txt`, and achieved `99.85%` Test Accuracy (`TN=635, FP=0, FN=1, TP=39`, `98.73%` F1) in `72,352 B`.

### 2.2 Detailed Paradigm Breakdown: Approach 2 (`Approch-2-DL/`)

Approach 2 ingests the exact serialized numpy splits (`X_train.npy`, `X_val.npy`, `X_test.npy`) produced by Approach 1 to eliminate split-seed confounding, testing five PyTorch architectures under a uniform optimization protocol (`AdamW(lr=1e-3, weight_decay=1e-4)`, `batch_size=128`, `BCEWithLogitsLoss(pos_weight=2961/189)`, early stopping on validation PR-AUC with `patience=15`):

1. **`MLP` (Baseline Feed-Forward Network):** Pure dense linear-ReLU-Dropout hierarchy (`29 → 64 → 32 → 1`, `p_drop=0.2`, `4,033` trainable parameters, **no BatchNorm layers**). Establishes neural baseline `95.38%` PR-AUC and `90.70%` F1 (`7` false positives, `1` false negative).
2. **`ResidualMLP` (Skip-Connected Tabular Network):** Introduces identity residual shortcuts across two `64`-unit hidden blocks (`18,625` parameters), improving Recall to `100.00%` and PR-AUC to `95.61%` at the expense of `10` false positives (`F1 88.89%`).
3. **`CNN1D` (1D Feature-Neighborhood Convolutional Network):** Treats the aligned `29`-element vector as a 1D signal `(B, 1, 29)`, applying two strided/padded convolution filters (`1→8→16`, `kernel_size=3`) before dense readout (`15,345` parameters). Achieves highest Approach 2 Test F1 (`90.91%`), `100.00%` Recall, and `98.82%` PR-AUC with sub-millisecond batch timing (`0.766 ms`).
4. **`TabTransformer` (Column-Wise Self-Attention Network):** Projects each of the `29` scalar coordinates independently via a shared linear embedding (`Linear(1, 32)`) into `29` tokens of width `32`, processes pairwise dependencies through `2` multi-head self-attention layers (`nhead=4`, `dim_feedforward=128`), flattens `(29 * 32 = 928)` embeddings, and classifies via `928 → 64 → 1`. Reaches top Deep Learning Test ROC-AUC (`99.94%`) and PR-AUC (`99.11%`), while incurring `~25×` batch inference latency (`18.647 ms`) and highest checkpoint weight (`351,993 B`).
5. **`Autoencoder` (Unsupervised One-Class Anomaly Detector):** Symmetric compression architecture (`29 → 32 → 10 → 32 → 29`) optimized strictly on benign training data (`y == 0`, `N = 2,961`) using `MSELoss` (`Adam(lr=1e-3)`). Flags claims whose reconstruction MSE exceeds the 95th validation percentile (`0.12858837842941284`). Fails empirically (`46.02%` ROC-AUC, `5.23%` PR-AUC, `1 / 40` frauds caught), furnishing vital scientific evidence that benign reconstruction error cannot isolate contextual amount fraud overlapping legitimate variance.

### 2.3 Detailed Paradigm Breakdown: Approach 3 (`Approach-3-Agentic-AI/` & Full-Stack Copilot)

Approach 3 shifts abstraction from numerical fitting to **workflow choreography, human readability, and audit compliance**:

* **Dual Codebase Presence:** Exists simultaneously as a standalone Python FastAPI prototype on port `:8020` (`Approach-3-Agentic-AI/app/`) and as an in-process TypeScript shadow endpoint inside the Express Node gateway (`backend/node-api/src/agentic/predict.ts`) paired with React dashboard components (`frontend/src/pages/AgenticPage.tsx` plus unmounted modular widgets in `frontend/src/agentic/`) and a Google Gemini conversational analyst route (`backend/node-api/src/routes/chat.ts` using `gemini-3.8-flash`).
* **Architectural Audit Reality:** Both Python `:8020` and Node `:3000` implementations run an analytical heuristic rule (`min(max((claim_amount * 0.01) + (patient_age * 0.005) - 2.0, 0.0), 1.0)`) and return static feature weights (`XAIExplainer`) alongside a 4-stage trace and truncated 12-character SHA-256 `audit_log` digest, decoupled from Approach 1 & 2 trained weights. Volume 05 analyzes both its working orchestration design and exact remediation steps to bridge trained empirical models into the agentic chain.

---

## 3. Six-Pillar Literature Foundations & Empirical Validation Map

The project's methodology synthesizes `20` literature references partitioned into `6` research pillars inside [`docs/01_LITERATURE_SURVEY.md`](../01_LITERATURE_SURVEY.md). Below is how each theoretical research pillar translates directly into code implementation and empirical confirmation or refutation inside this workspace:

| Research Pillar | Core Theoretical Premise | Concrete Implementation in Codebase | Empirical Outcome Observed in Project |
| :--- | :--- | :--- | :--- |
| **1. Cost-Sensitive Learning & PR-AUC under Skew** | Weighted objectives compensate for rare positive gradients; Precision-Recall curves isolate minority false alarms unlike ROC. | `class_weight='balanced'`, `scale_pos_weight=2961/189`, `BCEWithLogitsLoss(pos_weight)`, `average_precision_score` selection. | **Confirmed:** Prevents majority collapse; exposes `LR` precision drop (`62.50%`) that `99.50%` ROC-AUC otherwise conceals. |
| **2. Tree Ensembles on Heterogeneous Tabular Data** | Axis-aligned ensembles naturally isolate sharp conditional thresholds (`ClaimAmount` boundaries) without smooth interpolation overshoot. | `RandomForestClassifier(300)`, `XGBClassifier`, `HistGradientBoostingClassifier` in Track A & Track B. | **Confirmed:** Trees dominate both Test PR-AUC (`97.65%–100.00%`) and post-`2024-03-11` temporal generalization (`1.0000` AP). |
| **3. Tabular Deep Learning (`MLP`, `ResMLP`, `CNN1D`, `Transformer`)** | Skip connections, local convolution filters, and inter-column self-attention enable neural networks to approximate tabular manifolds. | 4 supervised PyTorch architectures in `Approch-2-DL/scripts/` trained with `AdamW` & PR-AUC early stopping. | **Nuanced Confirmation:** Reach `100%` Recall & up to `99.11%` PR-AUC on random splits, yet fall `8–20` AP points behind trees under temporal shift. |
| **4. Unsupervised Autoencoder Anomaly Detection** | Trained exclusively on normal samples, low-dimensional bottlenecks fail to reconstruct out-of-distribution fraud patterns. | `Autoencoder` (`29→32→10→32→29`) trained on `y==0` (`N=2,961`), 95th percentile validation threshold `0.1286`. | **Empirical Refutation on This Dataset:** Attains `5.23%` PR-AUC (`TP=1, FN=39`); frauds lie inside legitimate amount support (`\$7.4k–\$9.9k` vs `\$0.1k–\$10k`). |
| **5. Data Leakage, Feature Ablation & Temporal Integrity** | Target leakage (`ClaimStatus`), batch segmentation artifacts (`Cluster`), and random split leakage inflate benchmark metrics artificially. | `dataset_audit.json` forensics + `submission/` 4-scenario matrix (`baseline`, `intake`, `no_amount`, `temporal`). | **Confirmed:** Exposes `98.89%` fraud concentration in `Cluster==1` and proves `AP` collapse (`0.1869–0.2404`) when `ClaimAmount` is removed. |
| **6. Explainable Multi-Agent Workflows & Deployment Parity** | Real-world deployment needs auditable pipelines, feature explanations, interactive copilots, and bit-exact inference parity across runtimes. | `Approach-3-Agentic-AI/`, `backend/node-api/` (`gemini-3.8-flash`), `submission/scripts/export_browser.py` & `verify.py`. | **Split Result:** Browser parity verified to `4.42e-7` over `10,800` checks; Agentic pipeline proves structural UX but uses stub scoring (`\$250` saturation). |

---

## 4. Dual-Track Governance Model (`Track A` vs. `Track B`)

As repository development transitioned from isolated research scripts to an integrated web platform and finally into an audited academic benchmark, a critical tension emerged: *how to run systematic, strictly controlled scientific experiments (removing `Cluster`, enforcing deterministic PyTorch execution, testing temporal splits, and exporting pure-JS weights) without corrupting or overwriting teammate code and live platform models.*

The resolution (`ADR-01` & `ADR-06` in `DECISIONS.md`) established **Dual-Track Governance**:

```text
+---------------------------------------------------------------------------------------------------+
|                             DUAL-TRACK GOVERNANCE & MUTATION BOUNDARY                             |
+-------------------------------------------------+-------------------------------------------------+
|        TRACK A: EXPLORATORY & LIVE APP          |        TRACK B: CONTROLLED SCIENTIFIC LAB       |
+-------------------------------------------------+-------------------------------------------------+
| Directories:                                    | Directory:                                      |
|  - Approch-1/         (Classical Training)      |  - submission/                                  |
|  - Approch-2-DL/      (PyTorch Training)        |      - scripts/  (common, run_pipeline, export, |
|  - Approach-3-Agentic-AI/ (:8020 Prototype)     |                   verify, serve, make_figures)  |
|  - ml-service/        (:8000 FastAPI Inference) |      - models/   (per-scenario JSON & weights)  |
|  - backend/node-api/  (:3000 Express Gateway)   |      - results/  (audits, parity, 128-row CSV)  |
|  - frontend/          (:5173 React SPA)         |      - website/  (pure-JS static bundle)        |
+-------------------------------------------------+-------------------------------------------------+
| Write Policy:                                   | Write Policy:                                   |
|  Treated as READ-ONLY upstream by Track B.      |  Fully reproducible via run_pipeline.py;        |
|  Preserves original teammate folder spellings   |  exports immutable manifest + verification JSON |
|  and exact artifact metrics for presentation.   |  + Chromium parity suite.                       |
+-------------------------------------------------+-------------------------------------------------+
| Modeling Mechanics:                             | Modeling Mechanics:                             |
|  - Single 29f split incl. Cluster               |  - 4 Scenarios (baseline, intake, no_amount,    |
|  - Non-deterministic fast PyTorch execution     |    temporal) x 8 Models = 32 runs               |
|  - 0.5 fixed decision threshold (or AE 95th %)  |  - torch.use_deterministic_algorithms(True)     |
|  - HistGradientBoosting selected via Val F1=1.0 |  - Validation-F1 grid threshold selection       |
|  - XGBoost Test PR-AUC = 100.00%                |  - intake RandomForest selected (thr=0.4167)    |
+-------------------------------------------------+-------------------------------------------------+
```

### 4.1 Explicit Behavioral Differences Between Track A and Track B

Auditors inspecting both tracks must recognize four exact algorithmic differences verified in source files (`train_all_models.py` / `dl_common.py` vs. `submission/scripts/run_pipeline.py`) so numbers are never conflated across tracks:

| Algorithmic Parameter | Track A (`Approch-1/` & `Approch-2-DL/`) | Track B (`submission/scripts/run_pipeline.py`) | Why the Tracks Diverge |
| :--- | :--- | :--- | :--- |
| **`SVC` Probability Method** | `SVC(probability=True)` (5-fold Platt scaling internal calibration) | `SVC(probability=False)` + analytic `scipy.special.expit(decision_function)` | Eliminates Track A's internal Platt mismatch (`51` vs `39` flagged) and enables clean pure-JS export. |
| **`XGBClassifier` Configuration** | `use_label_encoder=False, eval_metric='logloss', scale_pos_weight=2961/189` (default `100` trees, `max_depth=6`, `lr=0.3`) | Explicit `XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, scale_pos_weight=..., eval_metric='logloss')` + scenario-specific imbalance ratio | Standardized explicitly in `run_pipeline.py` across random (`2961/189`) and temporal (`2953/199`) training splits. |
| **`HistGradientBoostingClassifier`** | Saved as `GradientBoosting.joblib`; selected as Track A best (`Val_F1=1.0`, `Test_F1=98.73%`) | `HistGradientBoostingClassifier(max_iter=100, class_weight='balanced', random_state=42)` | Produces `100.00%` Test AP across `baseline`, `intake`, and `temporal` under deterministic binning. |
| **PyTorch Execution Mode** | Standard PyTorch training (`torch.manual_seed(42)`) -> `MLP` (`95.38%`), `ResidualMLP` (`95.61%`), `CNN1D` (`98.82%`) | `torch.use_deterministic_algorithms(True)` -> e.g. `baseline_ResidualMLP` epoch `42` (`4.42s`, Test PR-AUC `95.67%`) vs Track A epoch `20` (`95.61%`) | Guarantees deterministic bitwise reproducibility at minor optimization trajectory shift. |
| **Operating Decision Threshold** | Fixed `0.5` probability (`model.predict()` / `probs >= 0.5`) | Per-run threshold selected strictly on Validation PR/F1 grid (e.g., `intake` `RandomForest` at `0.416667`) | Prevents arbitrary `0.5` mismatch under cost-sensitive `pos_weight` scaling. |
| **Deployed Winner Choice** | `GradientBoosting` (`HistGradientBoostingClassifier`, `29` features) in Track A report | `intake` `RandomForest` (`28` features, ex-`Cluster`) in `submission/results/selection.json` | Separates retrospective leaderboard performance from real-world pre-clustering intake safety. |

---

## 5. Complete Architectural Decision Records (`DECISIONS.md` + Platform ADRs)

All `13` design choices crystallized in [`DECISIONS.md`](../../DECISIONS.md) and operationalized across `ml-service/`, `backend/`, `frontend/`, and `submission/` are documented below with formal consequences and verification status:

| ID | Decision Summary | Technical Specification | Architectural Consequence | Verification Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **ADR-01** | Preserve Upstream Research Directories | Do not rename `Approch-1` / `Approch-2-DL` spelling or overwrite trained artifacts inside them. | Prevents broken relative paths (`models/*.joblib`, `Approch-2-DL/models/*.pt`) in `ml-service` and documentation references. | Workspace layout intact (`steps/5/output.txt`). |
| **ADR-02** | Standard `.venv` Environments | Isolate Python `3.11.x` dependencies (`requirements.txt`) per subsystem. | Separates research plotting packages (`seaborn`, `matplotlib`) from lean serving/submission dependencies where appropriate. | `training_metrics.json` records `Python 3.11.2`, `sklearn 1.9.1`. |
| **ADR-03** | Canonical 29-Feature Contract | Lock scalar order (`[0..6]`) and one-hot ordering (`[7..28]`) in `feature_names.txt` and `contract_metadata.json`. | Ensures live single-claim JSON transformation matches offline `.npy` training arrays within `atol=1e-5`. | `submission/results/dataset_audit.json` confirms `legacy_arrays_match: true`. |
| **ADR-04** | PR-AUC Primary Metric | Use `average_precision_score` and minority `F1` to rank models under `6.00%` prevalence. | Exposes weak false-positive control in `LogisticRegression`, `SVC`, and catastrophic failure of `Autoencoder`. | `model_results.json`, `*_metrics.json`, `metrics.csv`. |
| **ADR-05** | Explicit Uncalibrated Score Flag | Maintain `score_is_calibrated = False` consciousness across non-probabilistic outputs. | Clarifies that `BCEWithLogitsLoss(pos_weight=15.67)` outputs distorted odds and `Autoencoder` sigmoid scaling is monotonic anomaly normalization, not posterior fraud risk. | `ml-service/app/services/dl_service.py` & `submission/models/*/contract.json`. |
| **ADR-06** | Self-Contained `submission/` Track B | Mirror/re-generate clean experiments in `submission/` reading original Excel only. | Guarantees clean automated audit trail from raw XLSX hash (`952b6e23...`) to metrics CSV and browser demo. | `submission/results/run_manifest.json` & `verification.json`. |
| **ADR-07** | Exclusion of `Cluster` in Intake Mode | Define explicit `intake` scenario omitting `Cluster` from training and scoring. | Eliminates reliance on dataset-level batch clustering (`Cluster==1` holding `98.89%` of frauds) at single-claim inference time. | `submission/models/intake/contract.json` & `selection.json`. |
| **ADR-08** | Zero-Dependency Browser JS Export (`:8080`) | Export neural weights, trees, linear arrays, and scalers into pure JS executed by static web server `serve.py`. | Eliminates backend runtime requirement for portable static demo while mathematically proving inference parity. | `browser_parity.json`: `1,350` preproc + `10,800` scores matched. |
| **ADR-09** | Strict Pydantic Validation (`extra="forbid"`) | Forbid undeclared keys in FastAPI `ClaimRequest` (`claims.py`). | Prevents silent schema drift (e.g., camelCase vs PascalCase mismatches) from silently passing zero-filled features into predictions. | `verification.json` checks `4` negative HTTP `422` tests; `smoke_test.py`. |
| **ADR-10** | 3-Tier FastAPI + Express + React Topology | Serve Python ML (`:8000`) behind Express TypeScript API (`:3000`) and Vite React SPA (`:5173`). | Separates heavy numerical runtime (`PyTorch`/`scikit-learn`/`XGBoost`) from Node TypeScript web middleware & Gemini SDK calls. | Verified across `ml-service/`, `backend/node-api/`, `frontend/`. |
| **ADR-11** | Chronological Out-of-Time Holdout | Split test records strictly from `2024-03-11` onward (`N=671`, `37` frauds) in `temporal` scenario. | Reveals superior temporal stability of orthogonal decision trees (`1.0000` AP) over continuous neural layers (`0.7979–0.9172` AP). | `submission/results/metrics.csv` (`temporal` slice). |
| **ADR-12** | Validation-Only Threshold Selection | Sweep decision thresholds exclusively on validation probabilities (`N=675`) maximizing F1. | Yields unbiased operating cutoffs (`RF=0.4167`) checked against `1,000` bootstrap replicates (`bootstrap.json`). | `submission/results/selection.json` & `bootstrap.json`. |
| **ADR-13** | Graceful Frontend Offline Degradation | Provide deterministic fallback predictions in `ClassicalMLPage.tsx` (`AgenticPage.tsx` local fallback) when API servers are unreachable. | Enables UI inspection offline without crashes, while demanding clear documentation of synthetic fallback bounds (`ClassicalMLPage` returning `0.01` under normal claim sizes). | Verified in `frontend/src/pages/ClassicalMLPage.tsx` & `AgenticPage.tsx`. |

---

## 6. Three-Tier Platform Boundary & Communication Design

The live integrated system separates inference computation, gateway orchestration, and interactive browser presentation across three isolated processes plus an optional fourth standalone prototype:

### 6.1 Tier 1 — Python ML & DL Inference Microservice (`ml-service/`, Port `:8000`)

* **Framework & Runtime:** FastAPI + Uvicorn executing inside Python `.venv`.
* **State & Lifecycle:** Loads `Approch-1/models/preprocessor.joblib`, `Approch-1/data/processed/feature_names.txt`, all `5` classical `.joblib` estimators (`classical_service.py`), `Approch-2-DL/models/*_best.pt` weights + `Approch-2-DL/results/*_metrics.json` thresholds (`dl_service.py`, `mlp_service.py`).
* **Contract Defense:** `ClaimRequest` (`app/schemas/claims.py`) enforces exact `PascalCase` raw domain keys and raises HTTP `422` if unexpected attributes appear (`extra="forbid"`).
* **Endpoints Exposed:** `POST /predict/ml` (routes to selected Classical ML estimator), `POST /predict/dl` (routes to selected PyTorch architecture), and `POST /predict` (unified multi-model or default route).

### 6.2 Tier 2 — Express TypeScript API Gateway (`backend/node-api/`, Port `:3000`)

* **Framework & Runtime:** Node.js + Express (`tsx src/server.ts`) acting as the single CORS-enabled browser origin API facade.
* **Three Behavioral Planes:**
  1. **Transparent ML/DL Proxy (`src/routes/ml.ts`):** Forwards `/api/ml/*` payloads to `http://127.0.0.1:8000`, normalizing upstream availability failures cleanly.
  2. **In-Process Agentic Shadow Engine (`src/agentic/predict.ts`):** Handles `/api/agentic/predict` directly in TypeScript without calling `:8020` or `:8000`, running the linear threshold formula `AgenticEnsemble-v1`, emitting a 4-stage trace, and returning static feature weights.
  3. **Generative AI Analyst Copilot (`src/routes/chat.ts`):** Loads `GEMINI_API_KEY` from `../../Approach-3-Agentic-AI/.env` via `dotenv`, instantiates `GoogleGenerativeAI` with model **`gemini-3.8-flash`**, and injects current UI prediction state as system prompt context.

### 6.3 Tier 3 — React 19 + Vite Presentation Tier (`frontend/`, Port `:5173`)

* **Framework & Stack:** React `19` + TypeScript + Tailwind CSS + Vite dev server (`vite.config.ts`).
* **Mounted vs. Unmounted Architecture Audit (`src/App.tsx`):**
  * **Mounted Core Routes:** Renders navigation switching strictly among `HomePage` (`overview`), `ClassicalMLPage` (`ml`), `DLPage` (`dl`), and `AgenticPage` (`agentic`), conditionally mounting the sliding `ChatBox` copilot exclusively when `activeTab === "agentic"`.
  * **Unmounted Legacy/Modular Sibling Modules:** `frontend/src/agentic/` preserves self-contained `AgenticDashboard.tsx`, `ClaimInputForm.tsx`, `AgentCard.tsx`, `AuditTrail.tsx`, and `ScoreGauge.tsx` alongside `frontend/src/components/{PipelinePreview.tsx, ThemeToggle.tsx}` and `frontend/src/lib/{useApiHealth.ts, useScrollReveal.ts, useTheme.ts}`. Documenting this exact live import boundary prevents developer confusion when modifying the Agentic UI.

---

## 7. Engineering Evolution, Ground-Truth Verification & Structural Risk Register

### 7.1 Ground-Truth Live Serving Parity Spot Checks

Per [`PROJECT_PROGRESS.md`](../../PROJECT_PROGRESS.md), end-to-end integration across Tiers `1 → 2 → 3` was explicitly validated by passing concrete claims extracted directly from the raw workbook through live REST inference endpoints and checking prediction alignment against ground truth labels:

* **Legitimate Spot-Check (`Index 2` in workbook records):** Verified through the live REST inference chain to confirm negative fraud scoring and aligned preprocessing without schema transposition.
* **Fraud Spot-Check (`Index 0` in workbook records):** Verified through the live REST inference chain to confirm positive high-risk classification across Approach 1 and supervised Approach 2 models.

### 7.2 Cross-System Structural Risk & Technical Debt Register

An exhaustive code audit identifies seven structural issues and engineering tradeoffs documented across this suite (expanded surgically in Volumes `03`–`08`):

| Risk / Defect ID | Subsystem Affected | Concrete Verified Finding | Impact Severity & Volume Reference |
| :--- | :--- | :--- | :--- |
| **RISK-01** | `Approch-1/models/SVM.joblib` | `SVC(probability=True)` Platt curve disagrees with `predict()` hyperplane (`39` vs `51` test positives at `0.5` cutoff). | **Medium** — Requires unified threshold semantics across Track A (`Vol. 03`). |
| **RISK-02** | `Approch-1/scripts/create_notebooks.py` & `project/` paths | `preprocess.py` & `train_all_models.py` write to `project/...` requiring a Windows NTFS junction; notebook builder emits uninterpolated literal `{class_def}`. | **Medium (DevOps)** — Breaks clean checkout execution without Windows setup steps (`Vol. 03`). |
| **RISK-03** | `Approch-2-DL/` (`Autoencoder`) | Benign reconstruction MSE achieves `46.02%` ROC-AUC / `5.23%` PR-AUC (`TP=1, FN=39`), unusable as a standalone detector. | **Scientific Finding** — Retained intentionally as an instructive negative baseline (`Vol. 04`). |
| **RISK-04** | `Approach-3-Agentic-AI/` & `node-api/src/agentic/predict.ts` | Active scorer saturates to `1.0` for `claim_amount >= \$250` while workbook frauds begin at `\$7,465.07`; `run_agentic.py` fails to import router cleanly (`include_router(predict_fraud)`), `predict.py` duplicates `router = APIRouter(...)` and imports invalid `BackgroundTask`. | **High (Functional Stub)** — Must swap rule formula for Approach 1/2 ML inference before production (`Vol. 05`). |
| **RISK-05** | `frontend/src/pages/ClassicalMLPage.tsx` | Offline demo fallback formula `(ClaimAmount * 0.00003) + (PatientAge * 0.001) - 1.5` clamps to `0.01` unless `ClaimAmount > \$47,033`. | **Low (Offline Fallback Only)** — Harmless when `:3000`/`:8000` online; misleading in pure offline demo mode (`Vol. 06`). |
| **RISK-06** | `submission/` (`baseline` vs `intake` vs `temporal`) | `Cluster` concentrates `98.89%` of frauds; `no_amount` crashes AP to `0.1869–0.2404`; `temporal` degrades DL models (`0.7979–0.9172`). | **Critical Domain Insight** — Mandates tree ensembles + `intake` policy for deployment (`Vol. 07`). |
| **RISK-07** | `docs/01`–`05` & `ALL_THREE_APPROACHES_DOCUMENTATION.md` | Prior docs misstate Autoencoder widths (`16/8/16`), falsely claim `BatchNorm1d` on `MLP`, list stale `gemini-2.0-flash`, inflate `GradientBoosting.joblib` size by `10x`, and reference unmounted/non-existent UI files. | **Documentation Debt (Resolved Here)** — Explicitly reconciled in `Vol. 08` Discrepancy Ledger. |

---

## 8. Cross-Volume Dependency Map

Readers navigating deep implementation modules can follow this cross-reference matrix:

* **Need formal risk equations, exact dataset cardinality tables, `Cluster` crosstab proof, or the 29-column feature map?** → Go to [**Volume 02**](./02_MATHEMATICAL_FORMULATION_DATA_AUDIT_AND_CONTRACTS.md).
* **Need Classical ML estimator math, confusion matrices, Windows junction scripts, or `SVC` Platt-scaling proofs?** → Go to [**Volume 03**](./03_APPROACH_1_CLASSICAL_MACHINE_LEARNING.md).
* **Need exact PyTorch `nn.Module` tensor dimensions, parameter counts, training trajectories, or Autoencoder failure geometry?** → Go to [**Volume 04**](./04_APPROACH_2_DEEP_LEARNING.md).
* **Need Agentic orchestration traces, `\$250` threshold saturation analysis, `XAIExplainer` forensics, or `gemini-3.8-flash` setup?** → Go to [**Volume 05**](./05_APPROACH_3_AGENTIC_AI_XAI_AND_COPILOT.md).
* **Need FastAPI `:8000` / Node `:3000` / React `:5173` routes, Pydantic contracts, and UI inspection?** → Go to [**Volume 06**](./06_FULL_STACK_THREE_TIER_PLATFORM.md).
* **Need Track B `32`-run ablation benchmarks, bootstrap confidence intervals, `Row 1124` error forensics, and Chromium pure-JS parity?** → Go to [**Volume 07**](./07_TRACK_B_CONTROLLED_BENCHMARK_AND_BROWSER_LAB.md).
* **Need side-by-side tri-approach decision selection, full file/line inventory, PowerShell runbooks, or prior-docs audit?** → Go to [**Volume 08**](./08_CROSS_APPROACH_SYNTHESIS_INVENTORY_RUNBOOKS_AND_AUDIT.md).
