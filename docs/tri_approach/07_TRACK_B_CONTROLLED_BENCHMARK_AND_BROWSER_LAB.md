# Volume 07 — Track B Controlled Benchmark, Ablation & Pure-JS Browser Lab (`submission/`)

> **Document Classification:** Scientific Ablation Suite, Statistical Uncertainty & Edge Inference Parity Proof  
> **Primary Ownership:** **J. Ganesh Kumar Reddy (`23BDS024`)** (Track B benchmark & browser parity verification suite)  
> **Directory Scope:** [`submission/`](../../submission/) (`scripts/`, `models/`, `results/`, `START_DEMO.ps1`, `README.md`) — isolated completely under **ADR-06** (`DECISIONS.md`) so Track A (`Approch-1/`, `Approch-2-DL/`) remains untouched read-only upstream.  
> **Primary Ground-Truth Sources:** [`submission/README.md`](../../submission/README.md), [`submission/START_DEMO.ps1`](../../submission/START_DEMO.ps1), [`submission/scripts/common.py`](../../submission/scripts/common.py), [`run_pipeline.py`](../../submission/scripts/run_pipeline.py), [`export_browser.py`](../../submission/scripts/export_browser.py), [`verify.py`](../../submission/scripts/verify.py), [`serve.py`](../../submission/scripts/serve.py), [`make_figures.py`](../../submission/scripts/make_figures.py), [`evaluate_legacy.py`](../../submission/scripts/evaluate_legacy.py), [`check_browser.mjs`](../../submission/scripts/check_browser.mjs), [`submission/models/intake/contract.json`](../../submission/models/intake/contract.json), and [`submission/results/*.json` + `metrics.csv`](../../submission/results/verification.json).

---

## 1. Scientific Mandate of Track B (`submission/`)

Offline benchmarks in `Approch-1/` (`XGBoost PR-AUC = 100.00%`) and `Approch-2-DL/` (`TabTransformer PR-AUC = 99.11%`) demonstrate impressive static discrimination, yet leave **five rigorous scientific questions unanswered**:

1. **Pre-Clustering Intake Generalization (`intake` Scenario):** How severely do metrics degrade when post-hoc `Cluster` (`98.89%` fraud concentration in `Cluster == 1`) is stripped completely at inference time?
2. **Feature-Dominance Stress Test (`no_amount` Scenario):** Does demographic, temporal, or categorical context retain viable discrimination without `ClaimAmount`, or do classifiers rely essentially on upper-tail billing splits?
3. **Chronological Out-of-Time Drift (`temporal` Scenario):** Do continuous neural boundaries generalize as robustly as orthogonal decision trees across unseen calendar months (`>= 2024-03-11`), or does smooth extrapolation decay under time shift?
4. **Statistical Uncertainty Quantification (`bootstrap.json`):** Across a `675`-claim held-out test set containing only `40` fraud positives, how wide are the 95% confidence intervals around point estimates (`AP`, `F1`, `Recall`)?
5. **Reproducibility & Zero-Backend Deployment Parity (`export_browser.py` + `verify.py`):** Can deterministic model weights, standardization statistics, and threshold rules run identically inside pure browser JavaScript on `:8080` with proven machine-epsilon numeric parity against Python?

Track B isolates a clean, self-contained `32`-run matrix (`4` scenarios × `8` models = `128` structured rows inside `submission/results/metrics.csv`) that answers all five questions reproducibly from raw workbook digest `952b6e23...`.

---

## 2. Script Pipeline & Toolchain Inventory (`submission/`)

```text
Health Insurance Fraud Claims.xlsx (SHA-256: 952b6e...)
          |
          +----> [1] evaluate_legacy.py (31L) ---> results/legacy_artifact_audit.json (116L)
          |          Audits Approch-1 SVM .predict() [51] vs .predict_proba()>=0.5 [39]
          |
          +----> [2] run_pipeline.py (161L) -----> - 4 Scenarios x 8 Models (32 trained runs)
                     Uses common.py (63L)          - results/dataset_audit.json (134L)
                     + deterministic PyTorch       - results/metrics.csv (129L / 128 rows)
                                                   - results/selection.json (8L) [intake RF]
                                                   - results/analysis.json (147L) [Row 1124]
                                                   - results/bootstrap.json (20L) [B=1000]
                                                   - results/baseline_*_training.json
                                                   - models/intake/contract.json (59L)
                                                            |
          +-------------------------------------------------+
          |
          v
      [3] export_browser.py (62L) --------> Static pure-JS weights & weights payload under
          |                                 submission/website/dist/ + results/deployment.json
          +----> [4] make_figures.py (58L) -> Publication figures from Track B results
          |
          +----> [5] verify.py (48L) -------> - Validates 128 metric rows
                 + check_browser.mjs (21L)    - Verifies 32 API checks + 4 HTTP 422 checks
                 + serve.py (62L, :8080)      - Chromium headless parity check -> browser_parity.json:
                                                1,350 preproc (max_err=0) + 10,800 scores (4.42e-7)
```

### 2.1 Complete File Inventory Table (`submission/`)

| Script / Artifact | Lines | Engineering Function |
| :--- | :---: | :--- |
| `submission/README.md` | `100` | Controlled benchmark specification & local run instructions |
| `submission/START_DEMO.ps1` | `9` | Single-command PowerShell launcher spinning up static browser demo on `:8080` |
| `submission/scripts/common.py` | `63` | Shared workbook loader, SHA-256 verification, split logic, leakage exclusions (`ClaimStatus`) |
| `submission/scripts/run_pipeline.py` | `161` | Master `32`-run trainer/evaluator across `4` scenarios × `8` models; outputs metrics, selection, bootstrap, error analysis |
| `submission/scripts/export_browser.py` | `62` | Serializes scalers, categorical encodings, linear weights, tree structures, & neural matrices to pure JS targets under `submission/website/dist/` |
| `submission/scripts/make_figures.py` | `58` | Generates summary comparative charts directly from verified Track B CSV/JSON artifacts |
| `submission/scripts/evaluate_legacy.py` | `31` | Standalone auditor validating upstream Track A artifacts against canonical splits (`legacy_artifact_audit.json`) |
| `submission/scripts/verify.py` | `48` | End-to-end QA suite verifying `128` metric rows, `32` API requests, `4` negative `422` tests, `5` static routes, & JS parity |
| `submission/scripts/serve.py` | `62` | Pure-Python HTTP server exposing Track B static frontend bundle & artifacts on port `:8080` |
| `submission/scripts/check_browser.mjs` | `21` | Playwright/Chromium Node script executing live browser-side inference against Python ground truth (`browser_parity.json`) |

---

## 3. Experimental Matrix Design: 4 Scenarios × 8 Models (`32` Runs)

### 3.1 Scenario Axis (`4` Controlled Scientific Interventions)

| Scenario Identifier | Split Topology (`Train / Val / Test`) | Feature Policy Included / Dropped | Specific Scientific Hypothesis Tested |
| :--- | :--- | :--- | :--- |
| **`baseline`** | Random Stratified (`3150 / 675 / 675`, `seed=42`) | Canonical `29` features (`StandardScaler` + `OHE`, includes `Cluster` & `ClaimAmount`); `legacy_arrays_match: true` at `atol=1e-5` | Reproduces exact Track A array baseline under deterministic PyTorch & standardized `SVC`/`XGB`/`HGB` rules |
| **`intake`** | Random Stratified (`3150 / 675 / 675`, `seed=42`) | **Excludes `Cluster`** (`28` features; `models/intake/contract.json`) per **ADR-07** | Measures real-world deployment score when single unseen claims arrive prior to unsupervised clustering |
| **`no_amount`** | Random Stratified (`3150 / 675 / 675`, `seed=42`) | **Drops `ClaimAmount`** | Quantifies pure non-monetary residual signal in demographics, dates, specialties, and channels |
| **`temporal`** | Chronological Cutoff (`["2023-11-21", "2024-03-10"]`) → `3152 (199) / 677 (34) / 671 (37)` | Chronological holdout (`test >= 2024-03-11`, zero future leakage) | Benchmarks out-of-time robustness comparing axis-aligned trees vs. continuous deep nets & kernels |

### 3.2 Standardized Model Axis (`8` Deterministic Models in `run_pipeline.py`)

Notice key constructor distinctions verified directly in `submission/scripts/run_pipeline.py` (versus Track A's `train_all_models.py` and against erroneous hyperparameters cited in older markdown summaries):

| # | Model Identifier | Track B Constructor Verified in `run_pipeline.py` | Track A vs. Track B Parameter Difference |
| :---: | :--- | :--- | :--- |
| `1` | `LogisticRegression` | `LogisticRegression(max_iter=2000, ...)` | Consistent L2 solver ceiling (`2000` iters) |
| `2` | `RandomForest` | `RandomForestClassifier(n_estimators=300, ...)` | Identical `300` trees (`docs/03` wrongly cited `200`!) |
| `3` | `XGBoost` | `XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, ...)` | Explicitly pins XGBoost defaults across splits (`docs/03` wrongly cited `lr=0.05`) |
| `4` | `GradientBoosting` | `HistGradientBoostingClassifier(max_iter=100, ...)` | Explicit `max_iter=100` (`docs/03` wrongly cited `200`) |
| `5` | `SVM` | `SVC(..., probability=False)` → `scipy.special.expit(decision_function)` | Eliminates internal 5-fold Platt scaling discrepancy (`ADR` consistency & JS exportable) |
| `6` | `MLP` | `Approch-2-DL` `MLP` w/ **`torch.use_deterministic_algorithms(True)`** | Strict bitwise deterministic PyTorch kernels |
| `7` | `ResidualMLP` | `submission_models.ResidualMLP` w/ `use_deterministic_algorithms(True)` | Deterministic run peaked at **Epoch `42`** (`4.42s`, PR `95.67%`) vs Track A Epoch `20` |
| `8` | `CNN1D` | `submission_models.CNN1D` w/ `use_deterministic_algorithms(True)` | Deterministic local `k=3` 1D convolutions across feature indices |

*(Note: `TabTransformer` and `Autoencoder` from Track A are excluded from Track B's 8-model browser-exportable loop due to quadratic attention latency / JS transformer overhead and negative unsupervised baseline respectively.)*

---

## 4. Validation-Driven Model Selection (`selection.json`) & Statistical Uncertainty (`bootstrap.json`)

### 4.1 Production Intake Selection Winner (`selection.json`)

Under deployment realism rule **ADR-07** (`intake` scenario, omitting batch-computed `Cluster`) combined with strict validation-only threshold tuning **ADR-12**, Track B automatically selects **`RandomForest` (`intake` policy)** as the recommended production model:

| Metric / Selection Property | Verified Value (`selection.json` / `analysis.json`) | Interpretation |
| :--- | :---: | :--- |
| **Selected Policy Scenario** | `intake` (`ex-Cluster`, `28` features) | Safe for single-claim live intake screening |
| **Selected Estimator** | `RandomForest` (`n_estimators = 300`) | Superior calibration stability & false-positive zeroing |
| **Validation-Optimal Threshold** | **`0.4166666666666667`** (`5 / 12`) | Swept strictly on `N_val = 675` (`Val_AP = 1.0`, `Val_F1 = 1.0`) |
| **Held-Out Test AP (`PR-AUC`)** | **`1.000000`** (`100.00%`) | Perfect rank separation across all `40` test frauds vs `635` legits |
| **Held-Out Test `F1`** | **`0.987342`** (`98.73%`) | Precision = `100.00%` (`39/39`), Recall = `97.50%` (`39/40`) |
| **Held-Out Confusion Matrix** | `TN = 635`, `FP = 0`, `FN = 1`, `TP = 39` | Only `1` prediction error across all `675` test records |

### 4.2 1,000-Replicate Bootstrap 95% Confidence Intervals (`bootstrap.json`)

Point metrics on `40` positive test cases carry finite-sample uncertainty. Track B runs `B = 1,000` bootstrap resamples (`seed = 2026`) on the selected `intake` `RandomForest` test predictions (`N = 675`):

| Test Metric | Observed Point Estimate | Bootstrap 95% CI (`B = 1,000`, `seed = 2026`) | Interval Width | Statistical Stability Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Average Precision (`AP`)** | `1.0000` | **`[1.0000, 1.0000]`** | `0.0000` | Zero ranking variance across all `1,000` resamples! |
| **`F1` Score (`thr=0.4167`)** | `0.987342` | **`[0.9610, 1.0000]`** | `0.0390` | Tight upper-90s operational certainty |
| **Recall (`TPR`)** | `0.9750` | **`[0.9250, 1.0000]`** | `0.0750` | Reflects binary inclusion frequency of the single borderline `FN` (`Row 1124`) |

---

## 5. Surgical Error Forensics (`analysis.json`): `Row 1124` Post-Mortem

Because `intake` `RandomForest` makes literally **one classification error across `675` test records** (`TN=635, FP=0, TP=39, FN=1`) — while still scoring **`Test AP = 1.0000`** — `submission/results/analysis.json` (`147` lines) inspects that lone false negative microscopically:

| Forensic Attribute (`analysis.json`) | Exact Observed Value | Geometric / Analytical Explanation |
| :--- | :---: | :--- |
| **`source_row`** | **`1124`** | Exact workbook row index inside test split (`N = 675`) |
| **`target` (Ground Truth)** | **`1`** (`Fraud`) | True fraudulent claim missed at fixed validation threshold `0.4167` |
| **Predicted RF Probability (`score`)** | **`0.27`** (`27.00%`) | Scored strictly higher than every benign record (hence `AP = 1.0`!), yet below `0.416667`! |
| **`ClaimAmount`** | **`\$7,465.07`** | Equals the **exact global minimum fraud `ClaimAmount`** across the entire `4,500`-row dataset (`dataset_audit.json`)! |
| **`PatientAge`** | **`39`** | Prime adult demographic squarely overlapping typical legitimate claims |
| **`ClaimType`** | **`"Outpatient"`** | Common encounter category offering weaker conditional fraud separation |

**Remarkable Statistical Insight:** Every true negative among the `635` legitimate test claims scored strictly **below `0.27`** (giving `RandomForest` a perfect `1.0000` Average Precision!). The single error occurs exclusively because `Row 1124` sits on the **exact minimum floor of the fraud monetary distribution (`\$7,465.07`)**, receiving `0.27` voting proportion across trees — slightly beneath the validation-tuned threshold `0.4167` (`39/40` above vs `0/635` legitimate above). Lowering the operating threshold cleanly onto `0.27` recovers `40/40` recall with `0` false positives on this test fold (though validation selection rightly prohibits peeking at test probabilities).

---

## 6. Key Ablation Findings: `no_amount` Collapse & `temporal` Generalization Gap

### 6.1 Monetary Dependence Proof (`no_amount` Scenario)

Removing `ClaimAmount` completely strips away upper-tail billing boundaries (`\$7,465.07–\$9,978.43`) to test whether demographic, specialty, date, or channel features carry standalone signal:

| Model Family (`no_amount`) | Observed Held-Out Test `AP` (`metrics.csv`) | Collapse Magnitude vs. `baseline` (`~1.00`) | Scientific Verdict |
| :--- | :---: | :---: | :--- |
| **`GradientBoosting`** (`HGB`) | **`0.2404`** *(Highest in scenario)* | `-0.7596` | Retains weak `Cluster == 1` (`24.27%` baseline rate) signal |
| **`RandomForest`** | **`0.2066`** | `-0.7934` | Collapses by nearly `80` AP points |
| **`SVM`** (`RBF`) | **`0.1869`** | `-0.8131` | Essentially reduced to minority cluster density detection |
| **All Models Range** | **`0.1869 – 0.2404`** | Over `~76–81%` AP Loss | **Proves fraud detectability depends critically on interaction with `ClaimAmount`** |

### 6.2 Chronological Out-of-Time Benchmark (`temporal` Scenario: Test `>= 2024-03-11`, `N = 671`, `37` Frauds)

Unlike stratified random splits, chronological partitioning (`Train: <= 2023-11-20` `3152` claims, `Val: 2023-11-21..2024-03-10` `677` claims, `Test: >= 2024-03-11` `671` claims with `37` frauds) shifts normalized `ClaimYear` (`2024` unseen or skewed heavily toward test end of the `2022–2024` window!):

| Estimator (`temporal` Test, `N = 671`) | Architecture Family | Temporal Test `AP` (`PR-AUC`) | Temporal Test `F1` | Resilience Under `ClaimYear` Extrapolation |
| :--- | :--- | :---: | :---: | :--- |
| **Tree Ensembles (`RF`, `XGB`, `HGB`)** | Orthogonal Axis-Aligned Trees | **`1.0000`** | **`1.0000`** | **Invariant!** Zero degradation across time; splits cleanly ignore `ClaimYear` shift |
| **`CNN1D`** | 1D Feature Convolution + Dense | `0.9172` | `0.8108` | **Mild–Moderate Decay** (`-8.28` AP pts below trees; `k=3` convolves `ClaimYear, Month, DoW`) |
| **`LogisticRegression`** | Regularized Linear Hyperplane | `0.8799` | `0.7397` | **Moderate Decay** (`ClaimYear` weights shift intercept out of time) |
| **`MLP`** | 3-Layer Dense Neural Net | `0.8581` | `0.6957` | **Substantial Decay** (`-14.19` AP pts; dense continuous extrapolation errors on `2024`) |
| **`SVM`** (`RBF + expit`) | Radial Basis Margin Kernel | `0.8396` | `0.7619` | **Substantial Decay** (`ClaimYear` shift changes Euclidean distance to support vectors!) |
| **`ResidualMLP`** | Skip-Connected Highway MLP | **`0.7979`** | `0.7229` | **Severe Decay** (`-20.21` AP pts vs trees; residual skip propagates standardized `ClaimYear` shift) |

#### Why Continuous Kernels & Neural Nets Degrade While Trees Stay at `1.0000`

Think geometrically about `ClaimYear` (`StandardScaler` feature `idx 3`) under `temporal` splitting:

* Training covers `2022-07-09` through `2023-11-20` (centered mostly around `2022/2023`), so `2024` timestamps in Test (`>= 2024-03-11`) arrive as **uniformly shifted positive out-of-distribution values** (`z_year > +1.0σ` across every test claim!).
* **Orthogonal Tree Ensembles (`RF`, `XGB`, `HGB`)** learn thresholds on `ClaimAmount` and `Cluster` without wasting upper split depth on noisy `ClaimYear` drift, rendering predictions mathematically invariant to `ClaimYear = 2024` shift (**`1.0000` AP / `1.0000` F1**).
* Conversely, **smooth continuous predictors (`MLP`, `ResidualMLP`, `SVM`, `LogisticRegression`)** add linear/kernel weights `w_year * z_year` or Euclidean norms `||x - x_sv||` that shift monotonically across time, pushing benign `2024` high-value claims across classification boundaries unless `ClaimYear` is explicitly regularized or omitted!

---

## 7. Pure-JS Browser Inference Export & Parity Verification (`:8080`)

Rather than shipping heavy Python `scikit-learn` / `PyTorch` runtimes for static demo dissemination, `submission/scripts/export_browser.py` compiles fitted preprocessing statistics (`means`, `scales`, categorical index maps) and trained mathematical parameters into clean static JavaScript assets consumed by the static demo served via `serve.py` (`http://127.0.0.1:8080`, launched with `START_DEMO.ps1`), then audited independently by **`verify.py`** and headless Chromium script **`check_browser.mjs`**.

### 7.1 Exhaustive Verification Ledger (`browser_parity.json` & `verification.json`)

| Verification Suite | Scope Executed | Exact Count Verified | Observed Tolerance / Pass Result | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Array Parity (`dataset_audit.json`)** | Track B vs `Approch-2-DL/processed/*.npy` | `train`, `val`, `test` | Exact match within `atol = 1e-5` | **PASS** |
| **Metric Row Audit (`verification.json`)** | Recompute `metrics.csv` (`4` scenarios × `8` models) | **`128`** rows | Exact score recomputation verified | **PASS** |
| **Browser Preprocessing (`browser_parity.json`)** | JS feature transformer vs Python `preprocessor` (`val + test`) | **`1,350`** vectors | **`max_err = 0`** (`0.0` bit drift!) | **PASS** |
| **Browser Model Inference (`browser_parity.json`)** | JS model execution across `8` models vs Python (`val + test`) | **`10,800`** score evaluations | **`max_err = 4.4204131033342975e-7`**, `0` mismatches! | **PASS** |
| **API & Server Routing (`verification.json`)** | Live `serve.py` static & contract routes | **`32`** API checks, **`5`** static routes | `100%` HTTP `200` OK where valid | **PASS** |
| **Negative Input Defense (`verification.json`)** | Strict schema validation tests | **`4`** invalid payloads | Expected **`HTTP 422`** rejections | **PASS** |

---

## 8. Track B Execution Runbook (`Windows PowerShell`)

```powershell
# Quick Launch Static Verified Demo on Port :8080
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\submission
.\START_DEMO.ps1

# Or Run Complete Track B Pipeline + Parity Verification From Scratch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts/evaluate_legacy.py      # Audits Track A models -> legacy_artifact_audit.json
python scripts/run_pipeline.py         # Runs 32 controlled experiments -> metrics.csv, selection.json, analysis.json, bootstrap.json
python scripts/export_browser.py       # Compiles pure-JS weights -> deployment.json + website dist bundle
python scripts/make_figures.py         # Renders Track B comparative figures
python scripts/verify.py               # Verifies 128 rows, 32 API checks, 4x422 tests, & 10,800 Chromium JS scores!
```
