# Volume 09 — Model Cards: Approach 1 Classical Machine Learning Suite

> **Document Classification:** Standardized Engineering Model Card Portfolio (`Approch-1/` & `submission/`)  
> **Standard Sections per Model Card:** Model Identity & Storage → Mathematical Objective & Inductive Bias → Track A vs. Track B Constructor Spec → Quantitative Evaluation Profile → Derived Operating Diagnostics (`TN/FP/FN/TP`) → Efficiency & Latency Budget → Failure Modes, Calibration Caveats & Production Suitability.  
> **Primary Ground-Truth Sources:** [`Approch-1/scripts/train_all_models.py`](../../Approch-1/scripts/train_all_models.py), [`Approch-1/evaluation/model_results.json`](../../Approch-1/evaluation/model_results.json), [`Approch-1/logs/training_metrics.json`](../../Approch-1/logs/training_metrics.json), [`submission/scripts/run_pipeline.py`](../../submission/scripts/run_pipeline.py), [`submission/results/selection.json`](../../submission/results/selection.json), and [`submission/results/legacy_artifact_audit.json`](../../submission/results/legacy_artifact_audit.json).

---

## 1. Portfolio Comparison Overview

Every Classical ML estimator consumes the canonical `29`-feature standardized + one-hot numeric vector in Track A (`Approch-1/` + `ml-service/`) and scenario-governed vectors (`baseline`, `intake`, `no_amount`, `temporal`) in Track B (`submission/`). All five models preserve class-skew sensitivity across `3,150` training claims (`189` frauds / `2,961` legitimate, ratio `15.6667:1`).

| Model Card ID | Disk Key (`Approch-1/models/`) | True scikit-learn / XGBoost Class | Track A Status | Track B Status | Byte Footprint |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`CARD-ML-01`** | `LogisticRegression.joblib` | `sklearn.linear_model.LogisticRegression` | Interpretable Linear Baseline | Baseline + `temporal` (`0.8799` AP) | `1,103 B` |
| **`CARD-ML-02`** | `RandomForest.joblib` | `sklearn.ensemble.RandomForestClassifier` | `100.00%` Test Prec (`0` FP) | **`selection.json` Winner (`intake`)** | `3,465,961 B` |
| **`CARD-ML-03`** | `XGBoost.joblib` | `xgboost.XGBClassifier` | **Test PR-AUC Winner (`100.00%`)** | Top `1.0000` `temporal` tree (`100` trees) | `85,788 B` |
| **`CARD-ML-04`** | `SVM.joblib` | `sklearn.svm.SVC` | Exposes Platt `.predict` vs `proba` gap | `probability=False` + analytic `expit` | `96,459 B` |
| **`CARD-ML-05`** | `GradientBoosting.joblib` | `sklearn.ensemble.HistGradientBoostingClassifier` | **Selected `best_model.txt` (Val `100%`)** | Top `no_amount` AP (`0.2404`) + `temporal` `1.0` | `72,352 B` |

---

## 2. Model Card `CARD-ML-01`: `LogisticRegression`

### 2.1 Card Metadata & Intended Use

* **Estimator Class:** `sklearn.linear_model.LogisticRegression` (`Approch-1/scripts/train_all_models.py` L32–33)
* **Artifact Location:** `Approch-1/models/LogisticRegression.joblib` (`1,103` bytes)
* **Maintainer:** `J. Ganesh Kumar Reddy (23BDS024)`
* **Intended Role:** Fast, interpretable convex linear baseline quantifying what portion of health insurance fraud is linearly separable in the standardized `29`-feature coordinate space.
* **Out-of-Scope Role:** Standalone production claim blocking where false positives carry manual SIU review budgets (`37.50%` False Discovery Rate in Track A test evaluation).

### 2.2 Mathematical Objective & Exact Hyperparameters

Optimizes L2-regularized cost-sensitive log-loss over parameter vector $\mathbf{w} \in \mathbb{R}^{d}$ and intercept $b \in \mathbb{R}$ via quasi-Newton L-BFGS iterations (`max_iter=2000`):

$$\min_{\mathbf{w}, b} \; \frac{1}{2}\|\mathbf{w}\|_2^2 + \sum_{i=1}^{N_{\text{train}}} w_{y_i} \log\!\Big(1 + \exp\!\big(-(2y_i - 1)(\mathbf{w}^\top \mathbf{x}_i + b)\big)\Big)$$

| Parameter | Track A Verified Value | Track B Verified Value (`run_pipeline.py`) | Functional Rationale |
| :--- | :--- | :--- | :--- |
| **`max_iter`** | `2000` | `2000` | Prevents L-BFGS convergence warnings under collinear OHE + large scalar gradients |
| **`class_weight`** | `'balanced'` | `'balanced'` | Weights positive residuals $15.6667\times$ higher than negatives |
| **`solver`** | `'lbfgs'` | default (`'lbfgs'`) | Fast deterministic deterministic second-order quasi-Newton convergence |
| **`random_state`** | `42` | `42` | Seed consistency |

### 2.3 Quantitative Evaluation & Confusion Matrix Profile

* **Validation (`N = 675`, `41` Frauds):** `F1 = 84.21%`, `PR-AUC = 94.11%`.
* **Held-Out Test (`N = 675`, `40` Frauds, `635` Legitimate):**
  * Accuracy: `96.44%` | Precision: `62.50%` | Recall: **`100.00%`** | F1: `76.92%`
  * ROC-AUC: `99.50%` | PR-AUC: `91.86%`
  * Confusion Counts: **`TN = 611`**, **`FP = 24`**, **`FN = 0`**, **`TP = 40`**
* **Derived Confusion Diagnostics (computed from `(611, 24, 0, 40)`):**
  * Specificity (`TNR`): `611 / 635 = 96.22%` | False Positive Rate (`FPR`): `24 / 635 = 3.78%`
  * False Negative Rate (`FNR`): `0.00%` | False Discovery Rate (`FDR`): `24 / 64 = 37.50%`
  * Negative Predictive Value (`NPV`): **`100.00%`** | Matthews Correlation (`MCC`): **`0.7757`**
* **Track B Chronological Holdout (`temporal`, Test `N = 671`, `37` Frauds):** `AP = 0.8799`, `F1 = 0.7397` (`-12.01` AP pts below tree ensembles).

### 2.4 Compute Profile, Edge Geometry & Failure Analysis

* **Training / Inference / Memory:** Fit time `0.013s`; batch `100`-claim inference `0.0003s` (`3.0 µs/claim`); serialized disk weight `1,103 B`.
* **Why `24` False Positives Occur:** Because frauds occupy `Cluster == 1` (`98.89%`) and `ClaimAmount >= \$7,465.07`, positive weights `(w_amount > 0, w_cluster ~ positive)` tilt the linear hyperplane so all legitimate `Cluster == 1` claims exceeding roughly `\$7,500` cross into positive log-odds (`24` FP), whereas non-linear tree splits carve finer multi-feature interaction pockets around legitimate upper-tail claims.
* **Calibration Status:** `score_is_calibrated = False` due to `class_weight='balanced'` log-odds intercept upward shift ($+\ln 15.6667$).

---

## 3. Model Card `CARD-ML-02`: `RandomForestClassifier`

### 3.1 Card Metadata & Intended Use

* **Estimator Class:** `sklearn.ensemble.RandomForestClassifier` (`Approch-1/scripts/train_all_models.py` L34)
* **Artifact Location:** `Approch-1/models/RandomForest.joblib` (`3,465,961` bytes = `3.31 MB`)
* **Track B Role:** **Primary Selected Production Estimator (`intake` ex-`Cluster` Policy, `submission/results/selection.json`)** exported to pure JavaScript weights (`export_browser.py`).
* **Intended Role:** High-precision tabular fraud screener when zero-false-positive operational strictness and temporal stability (`>= 2024-03-11`) take precedence over disk compactness.

### 3.2 Structural Specification & Exact Hyperparameters

Averages `T = 300` randomized CART decision trees grown over stratified bootstrap replicates, weighting split node Gini impurity symmetrically by class inverse frequency (`w_1 / w_0 = 15.6667`):

$$\hat{p}_{\text{RF}}(\mathbf{x}) = \frac{1}{300} \sum_{t=1}^{300} \frac{w_1 \, n_{t,1}(\mathbf{x})}{w_0 \, n_{t,0}(\mathbf{x}) + w_1 \, n_{t,1}(\mathbf{x})}$$

| Parameter | Track A Verified Value | Track B Verified Value (`run_pipeline.py`) | Impact on Behavior |
| :--- | :--- | :--- | :--- |
| **`n_estimators`** | `300` | `300` | Smooths probability quantization (`1/300 = 0.0033` steps); note earlier `docs/03` wrongly cited `200` |
| **`class_weight`** | `'balanced'` | `'balanced'` | Reweights leaf vote counts & Gini reductions proportionally |
| **`random_state`** | `42` | `42` | Fixes bootstrap & sub-feature (`max_features='sqrt'`) draws |
| **`n_jobs`** | `-1` | `-1` / parallelized | Utilizes multi-core CPU parallelism during fit & batch scoring |

### 3.3 Quantitative Evaluation & Confusion Matrix Profile

* **Track A Validation (`N = 675`):** `F1 = 98.77%`, `PR-AUC = 100.00%`.
* **Track A Held-Out Test (`N = 675`, `40` Frauds):**
  * Accuracy: **`99.85%`** | Precision: **`100.00%`** | Recall: `97.50%` | F1: `98.73%`
  * ROC-AUC: **`99.996%`** | PR-AUC: **`99.94%`**
  * Confusion Counts: **`TN = 635`**, **`FP = 0`**, **`FN = 1`**, **`TP = 39`**
* **Derived Operating Ratios (`(635, 0, 1, 39)`):**
  * Specificity: **`100.00%`** | `FPR`: **`0.00%`** | `FNR`: `2.50%` | `FDR`: **`0.00%`** | `NPV`: `99.84%` | `MCC`: **`0.9867`**
* **Track B `intake` Selected Policy (`ex-Cluster`, `thr = 0.416667`):**
  * Validation `AP = 1.0000`, `F1 = 1.0000`
  * Test `AP = 1.0000`, `F1 = 0.987342`, `TN = 635, FP = 0, FN = 1, TP = 39`
  * `B = 1000` Bootstrap 95% CIs (`seed=2026`): `AP = [1.0, 1.0]`, `F1 = [0.9610, 1.0]`, `Recall = [0.9250, 1.0]`
* **Track B `no_amount` & `temporal`:** `no_amount` Test `AP = 0.2066`; `temporal` Test `AP = 1.0000`, `F1 = 1.0000`.

### 3.4 Compute Profile, `Row 1124` Failure & Tradeoffs

* **Training / Inference / Memory:** Slowest classical fit (`0.625s`), slowest classical batch scoring (`0.0537s` for `100` claims = `537 µs/claim`), largest disk artifact (`3,465,961 B`).
* **Single Error Signature:** Under Track B `intake` evaluation (`analysis.json`), its sole test error (`source_row = 1124`, fraud claim at `\$7,465.07`, age `39`, `Outpatient`) received tree vote proportion **`0.27`**, sitting right below threshold `0.4167` while remaining strictly higher than all `635` benign test records (`AP = 1.0`).

---

## 4. Model Card `CARD-ML-03`: `XGBClassifier` (`XGBoost`)

### 4.1 Card Metadata & Intended Use

* **Estimator Class:** `xgboost.XGBClassifier` (`Approch-1/scripts/train_all_models.py` L35–36)
* **Artifact Location:** `Approch-1/models/XGBoost.joblib` (`85,788` bytes = `83.78 KB`)
* **Headline Distinction:** **Highest Overall Track A Held-Out Test Performance** (`Test PR-AUC = 100.00%`, `Test ROC-AUC = 100.00%`, `Test F1 = 98.77%`, `MCC = 0.9870`), achieving near-optimal Pareto efficiency (`28 ms` training, `6 µs/claim` inference).

### 4.2 Mathematical Objective & Exact Hyperparameters

Fits additive gradient-boosted depth-bounded trees $\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i)$ minimizing weighted logistic deviance using exact/hist second-order Newton leaf steps ($g_i, h_i$):

$$\mathcal{L}^{(t)} = \sum_{i=1}^{N_{\text{train}}} \Big( w_i \, \big[ g_i f_t(\mathbf{x}_i) + \tfrac{1}{2} h_i f_t(\mathbf{x}_i)^2 \big] \Big) + \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} \omega_j^2, \qquad w_i = \begin{cases} \text{scale\_pos\_weight} & y_i = 1 \\ 1 & y_i = 0 \end{cases}$$

| Parameter | Track A Verified Value (`train_all_models.py`) | Track B Verified Value (`run_pipeline.py`) | Notes |
| :--- | :--- | :--- | :--- |
| **`n_estimators`** | library default (`100`) | explicit `100` | Earlier `docs/03` misstated learning rate; `run_pipeline.py` uses `100` |
| **`max_depth`** | library default (`6`) | explicit `6` | Enables `6`-level conjunctive splits across `ClaimAmount`, `Cluster`, demographics |
| **`learning_rate`** | library default (`0.3`) | explicit `0.3` | Note `docs/03` falsely claimed `lr=0.05` in Track B |
| **`scale_pos_weight`** | `2961 / 189` (`15.6667`) | scenario-dynamic (`2961/189` or `2953/199` temporal) | Scales minority gradients proportionally per split |
| **`eval_metric`** | `'logloss'` | `'logloss'` | Standard binary log-likelihood objective |
| **`use_label_encoder`** | `False` | deprecated-safe omission / explicit target | Clean binary `{0, 1}` targets |

### 4.3 Quantitative Evaluation & Confusion Matrix Profile

* **Track A Validation (`N = 675`):** `F1 = 98.77%`, `PR-AUC = 99.94%`.
* **Track A Held-Out Test (`N = 675`, `40` Frauds):**
  * Accuracy: **`99.85%`** | Precision: `97.56%` | Recall: **`100.00%`** | F1: **`98.77%`**
  * ROC-AUC: **`100.00%`** (`1.0000`) | PR-AUC: **`100.00%`** (`1.0000`)
  * Confusion Counts: **`TN = 634`**, **`FP = 1`**, **`FN = 0`**, **`TP = 40`**
* **Derived Operating Diagnostics (`(634, 1, 0, 40)`):**
  * Specificity: `99.84%` | `FPR`: `0.16%` | `FNR`: **`0.00%`** | `FDR`: `2.44%` | `NPV`: **`100.00%`** | `MCC`: **`0.9870`**
* **Track B Chronological Holdout (`temporal`, Test `N = 671`):** **`AP = 1.0000`**, **`F1 = 1.0000`**.

### 4.4 Compute Profile & Calibration Note

* **Training / Inference / Memory:** Fit time **`0.028s`** (`22.3×` faster than `RandomForest`); batch `100`-claim inference **`0.0006s`** (`6.0 µs/claim`, `89.5×` faster than `RandomForest`); `85,788` bytes (`40.4×` smaller than `RandomForest`).
* **Threshold Interpretation:** Although `Test PR-AUC = 1.0000` proves every true test fraud outranks every benign test claim continuously, the default `0.50` logit probability threshold lets `1` borderline legitimate case trigger (`FP = 1`), placing `scale_pos_weight` decision threshold marginally conservatively.

---

## 5. Model Card `CARD-ML-04`: `SVC` (`SVM` with RBF Kernel)

### 5.1 Card Metadata & Intended Use

* **Estimator Class:** `sklearn.svm.SVC` (`Approch-1/scripts/train_all_models.py` L37)
* **Artifact Location:** `Approch-1/models/SVM.joblib` (`96,459` bytes = `94.20 KB`)
* **Scientific Role:** Evaluates whether radial smooth distance boundaries ($K(x, x') = \exp(-\gamma \|x - x'\|^2)$) outperform axis-aligned orthogonal tree hyperplanes on mixed standardized continuous + binary tabular columns.

### 5.2 Exact Constructor Spec & Track A vs. Track B Calibration Shift

| Parameter | Track A Verified Value (`train_all_models.py`) | Track B Verified Value (`run_pipeline.py`) | Reason for Difference |
| :--- | :--- | :--- | :--- |
| **`kernel`** | `'rbf'` | `'rbf'` | Gaussian similarity radius in `29`-dim Euclidean space |
| **`class_weight`** | `'balanced'` | `'balanced'` | Asymmetric penalty $C_1 / C_0 = 15.6667$ |
| **`probability`** | **`True`** *(5-fold Platt sigmoid calibration)* | **`False`** *(uses `scipy.special.expit(decision_function)`)* | Resolves Track A `.predict()` vs `.predict_proba()` mismatch cleanly! |
| **`random_state`** | `42` | `42` | Reproducible internal CV folds when `probability=True` |

### 5.3 Quantitative Evaluation & Platt Discrepancy Profile

Because Track A invoked `model.predict(X_test)` for discrete confusion matrices yet `model.predict_proba(X_test)[:, 1]` for ROC/PR curves (and `ml-service` invokes `.predict_proba()`), **two distinct confusion matrices exist for the exact same `SVM.joblib` binary file** (`legacy_artifact_audit.json`):

* **Track A Validation (`N = 675`):** `F1 = 87.91%`, `PR-AUC = 94.42%`.
* **Track A Test — Raw Margin `.predict(X_test)` (`model_results.json`, `51` Positive Flags):**
  * Accuracy: `97.78%` | Precision: `74.51%` | Recall: `95.00%` | F1: `83.52%` | ROC: `99.55%` | PR: `92.52%`
  * Confusion Counts: **`TN = 622`**, **`FP = 13`**, **`FN = 2`**, **`TP = 38`**
  * Derived Rates: Specificity `97.95%`, `FPR = 2.05%`, `FNR = 5.00%`, `FDR = 25.49%`, `NPV = 99.68%`, `MCC = 0.8304`.
* **Track A Test — Platt `predict_proba >= 0.5` (`legacy_artifact_audit.json`, `39` Positive Flags, matches `ml-service`!):**
  * Accuracy: **`98.96%`** | Precision: **`92.31%`** | Recall: `90.00%` | F1: **`91.14%`**
  * Confusion Counts: **`TN = 632`**, **`FP = 3`**, **`FN = 4`**, **`TP = 36`**
  * Derived Rates: Specificity `99.53%`, `FPR = 0.47%`, `FNR = 10.00%`, `FDR = 7.69%`, `NPV = 99.37%`, `MCC = 0.9060`.
* **Track B Stress Runs:** Lowest AP in `no_amount` (`0.1869`), strong temporal decay in `temporal` (`0.8396` AP, `0.7619` F1 — Euclidean distance $\exp(-\gamma \|x - x_{sv}\|^2)$ suffers out-of-distribution `ClaimYear = 2024` shifts).

---

## 6. Model Card `CARD-ML-05`: `HistGradientBoostingClassifier` (`GradientBoosting`)

### 6.1 Card Metadata & Intended Use

* **Estimator Class:** `sklearn.ensemble.HistGradientBoostingClassifier` (`Approch-1/scripts/train_all_models.py` L38)
* **Artifact Location:** `Approch-1/models/GradientBoosting.joblib` (**`72,352` bytes** — note `05_results.md` misprint `723,352`)
* **Headline Distinction:** **Track A Automatically Selected Best Estimator (`evaluation/best_model.txt`)**, achieving `100.00%` Validation F1 (`0` errors on `N = 675` validation set) and `100.00%` Test Precision (`0` false positives) at compact `70.66 KB` footprint.

### 6.2 Exact Hyperparameters & Algorithmic Structure

Bins continuous attributes (`max_bins=255` default) into integer histograms to accelerate LightGBM-style leaf-wise gradient-boosted splits under `class_weight='balanced'`:

| Parameter | Track A Verified Value (`train_all_models.py`) | Track B Verified Value (`run_pipeline.py`) | Notes |
| :--- | :--- | :--- | :--- |
| **`class_weight`** | `'balanced'` | `'balanced'` | Inverse class weight gradient scaling |
| **`max_iter`** | library default (`100`) | explicit `100` | Number of additive histogram trees (`docs/03` misstated `200`) |
| **`random_state`** | `42` | `42` | Reproducible binning/split tie breaking |

### 6.3 Quantitative Evaluation & Confusion Matrix Profile

* **Track A Validation (`N = 675`):** **`F1 = 100.00%`**, **`PR-AUC = 100.00%`** (`0` errors — triggers `best_model.txt`).
* **Track A Held-Out Test (`N = 675`, `40` Frauds):**
  * Accuracy: **`99.85%`** | Precision: **`100.00%`** | Recall: `97.50%` | F1: `98.73%`
  * ROC-AUC: `98.47%` | PR-AUC: `97.65%`
  * Confusion Counts: **`TN = 635`**, **`FP = 0`**, **`FN = 1`**, **`TP = 39`**
* **Derived Operating Diagnostics (`(635, 0, 1, 39)`):**
  * Specificity: **`100.00%`** | `FPR`: **`0.00%`** | `FNR`: `2.50%` | `FDR`: **`0.00%`** | `NPV`: `99.84%` | `MCC`: **`0.9867`**
* **Track B Performance:** Highest residual `no_amount` AP (`0.2404`) and **`1.0000` AP / `1.0000` F1** on `temporal` split!

### 6.4 Compute Profile & Histogram Quantization Subtlety

* **Training / Inference / Memory:** Fit time `0.192s`; `100`-claim batch inference `0.0013s` (`13.0 µs/claim`, `41×` faster than `RandomForest`); compact **`72,352` bytes** (`48×` smaller than `RandomForest`).
* **ROC/PR Ranking Note:** Notice how `HistGradientBoosting` shares the identical discrete `0.5` confusion matrix `(635, 0, 1, 39)` as `RandomForest` on Track A Test, yet scores `97.65%` Test PR-AUC vs `RandomForest`'s `99.94%`: because integer histogram discretization maps its lone missed boundary fraud into a confident benign bin leaf rather than hovering near threshold. For threshold-ranking critical deployments without `Cluster`, Track B selects `RandomForest` (`intake AP = 1.0`).
