# Volume 03 — Approach 1: Classical Machine Learning Engineering (`Approch-1/`)

> **Document Classification:** Subsystem Engineering Reference & Model Card Suite  
> **Primary Ownership:** **J. Ganesh Kumar Reddy (`23BDS024`)**  
> **Workspace Path:** [`Approch-1/`](../../Approch-1/) *(legacy spelling preserved intentionally under `ADR-01`)*  
> **Verified Execution Environment (`training_metrics.json`):** `Python 3.11.2`, `scikit-learn 1.9.1`, `xgboost`, Run Timestamp `2026-09-13 04:44:42`, Total Training Pipeline Wall Clock `4.12s`.  
> **Primary Ground-Truth Sources:** [`Approch-1/README.md`](../../Approch-1/README.md), [`Approch-1/scripts/preprocess.py`](../../Approch-1/scripts/preprocess.py), [`Approch-1/scripts/train_all_models.py`](../../Approch-1/scripts/train_all_models.py), [`Approch-1/scripts/eda_plots.py`](../../Approch-1/scripts/eda_plots.py), [`Approch-1/scripts/comparison_plots.py`](../../Approch-1/scripts/comparison_plots.py), [`Approch-1/scripts/error_analysis.py`](../../Approch-1/scripts/error_analysis.py), [`Approch-1/scripts/create_notebooks.py`](../../Approch-1/scripts/create_notebooks.py), [`Approch-1/evaluation/model_results.json`](../../Approch-1/evaluation/model_results.json), [`Approch-1/logs/training_metrics.json`](../../Approch-1/logs/training_metrics.json), and [`Approch-1/documentation/`](../../Approch-1/documentation/06_setup_and_usage.md).

---

## 1. Subsystem Philosophy & Role Across the Project

Approach 1 establishes both the **canonical tabular data preprocessing pipeline** and the **high-precision tree/linear/kernel benchmark suite** for the entire repository. Its engineering influence extends far beyond standalone scripts:

1. **Upstream Data Contract Producer:** `Approch-1/scripts/preprocess.py` fits the canonical `StandardScaler` + `OneHotEncoder` `ColumnTransformer` (`models/preprocessor.joblib`), exports the ordered `29` feature identifiers (`data/processed/feature_names.txt`), and materializes the stratified `train` (`3,150`), `val` (`675`), and `test` (`675`) numpy arrays that Approach 2 consumes directly to guarantee split-seed isolation.
2. **Production Serving Weight Store:** All five serialized scikit-learn/XGBoost model files (`models/*.joblib`) alongside `preprocessor.joblib` are loaded directly at startup by `ml-service/app/services/classical_service.py` on port `:8000` to power the `ClassicalMLPage` web interface.
3. **Reference Baseline for Track B:** Track B's `submission/` ablation suite reproduces `Approch-1/`'s preprocessing matrices to within `atol = 1e-5` (`legacy_arrays_match: {train: true, val: true, test: true}` in `dataset_audit.json`) before testing removal of `Cluster` (`intake`), removal of `ClaimAmount` (`no_amount`), and chronological splits (`temporal`).

---

## 2. Directory Layout & Artifact Inventory (`Approch-1/`)

```text
Approch-1/
|-- README.md                           # Subsystem guide & pipeline overview (63 lines)
|-- requirements.txt                    # Python ML/visualization dependencies (26 lines)
|-- data/
|   |-- raw/
|   |   `-- Health Insurance Fraud Claims.xlsx
|   `-- processed/
|       |-- feature_names.txt           # Canonical 29-feature line-delimited manifest
|       |-- X_train.npy, y_train.npy    # (3150, 29), (3150,)
|       |-- X_val.npy,   y_val.npy      # (675, 29),  (675,)
|       `-- X_test.npy,  y_test.npy     # (675, 29),  (675,)
|-- scripts/
|   |-- eda_plots.py                    # Exploratory visualization suite (109 lines)
|   |-- preprocess.py                   # Data cleaning, splitting & 29f encoding (98 lines)
|   |-- train_all_models.py             # 5-estimator training & evaluation (148 lines)
|   |-- comparison_plots.py             # Cross-model ROC/PR/metric chart suite (72 lines)
|   |-- error_analysis.py               # False-positive / false-negative forensics (55 lines)
|   `-- create_notebooks.py             # Programmatic .ipynb generator (32 lines, contains template bug)
|-- models/
|   |-- preprocessor.joblib             # Fitted ColumnTransformer (StandardScaler + OneHotEncoder)
|   |-- LogisticRegression.joblib       # 1,103 bytes
|   |-- RandomForest.joblib             # 3,465,961 bytes (300 trees)
|   |-- XGBoost.joblib                  # 85,788 bytes
|   |-- SVM.joblib                      # 96,459 bytes (RBF SVC w/ probability=True)
|   `-- GradientBoosting.joblib         # 72,352 bytes (actually HistGradientBoostingClassifier!)
|-- evaluation/
|   |-- best_model.txt                  # Selected model key ("GradientBoosting", Val_F1=1.0000)
|   |-- model_results.json              # Complete Val + Test evaluation JSON (77 lines)
|   |-- result_analysis.md              # Auto/curated markdown summary (21 lines)
|   `-- result_analysis.txt             # Plaintext analysis summary (13 lines)
|-- logs/
|   `-- training_metrics.json           # Environment, timing, byte sizes & metrics log (125 lines)
`-- documentation/
    |-- 02_dataset_analysis.md          # Field descriptions & distributions (89 lines)
    |-- 04_methodology.md               # Preprocessing & modeling choices (65 lines)
    |-- 05_results.md                   # Recorded results writeup (contains 723,352 typo) (69 lines)
    |-- 06_setup_and_usage.md           # Windows NTFS junction instructions (133 lines)
    `-- 07_learning_notes.md            # Team engineering reflections (51 lines)
```

---

## 3. End-to-End Execution Pipeline Walkthrough (`scripts/*.py`)

The Approach 1 script suite executes sequentially across five analytical modules (`eda_plots.py` → `preprocess.py` → `train_all_models.py` → `comparison_plots.py` → `error_analysis.py`), plus an auxiliary notebook builder (`create_notebooks.py`).

```text
  [Health Insurance Fraud Claims.xlsx]
                  |
                  +-------------------------------+
                  |                               |
                  v                               v
       1. scripts/eda_plots.py         2. scripts/preprocess.py
       (109 lines)                     (98 lines)
       Generates class skew,           - Drops IDs + Diagnosis/Procedure/Location + ClaimStatus
       amount distribution &           - Extracts Year/Month/DayOfWeek from ClaimDate
       correlation visuals             - Stratified split (70/15/15, seed=42)
                                       - Fits ColumnTransformer on X_train ONLY
                                       - Saves preprocessor.joblib + feature_names.txt + .npy
                                                  |
                                                  v
                                       3. scripts/train_all_models.py
                                       (148 lines)
                                       - Trains 5 estimators on X_train (29f)
                                       - Evaluates Val (675) & Test (675)
                                       - Selects best_model.txt by Val_F1 (GradientBoosting)
                                       - Benchmarks 100-claim inference time & byte sizes
                                       - Emits models/*.joblib + evaluation/model_results.json
                                         + logs/training_metrics.json
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
              4. scripts/comparison_plots.py                    5. scripts/error_analysis.py
              (72 lines)                                        (55 lines)
              Generates cross-model                             Inspects FN / FP cases of
              ROC, PR, F1, Latency &                            selected best estimator &
              Confusion-Matrix figures                          writes result_analysis.{md,txt}
```

### 3.1 Windows Virtual Workspace Requirement (`project/` NTFS Junction)

A crucial filesystem subtlety documented in [`Approch-1/documentation/06_setup_and_usage.md`](../../Approch-1/documentation/06_setup_and_usage.md) affects re-running scripts directly from source:

* **Embedded Relative Prefix:** Both `preprocess.py` and `train_all_models.py` resolve paths relative to a parent subdirectory prefix `project/` (e.g., `project/data/raw/...`, `project/models/...`, `project/evaluation/...`) originating from the author's original project staging layout.
* **Verified Windows Workaround:** Because the directory on disk inside `mini-project` is `Approch-1/` (not `Approch-1/project/`), executing Python scripts directly from `Approch-1/` requires mapping a Windows directory junction before running scripts:
  ```powershell
  cd c:\Users\RAVIPRAKASH\Downloads\mini-project\Approch-1
  New-Item -ItemType Junction -Path ".\project" -Target "."
  ```
* **Engineering Lesson:** Creating this directory junction maps `./project/*` transparently to `.` without duplicating data or modifying immutable `Approch-1/scripts/*.py` source files.

### 3.2 Script-by-Script Functional Breakdown

#### `scripts/eda_plots.py` (`109` lines)
Loads `Health Insurance Fraud Claims.xlsx` cleanly prior to splitting, producing visual distributions confirming class imbalance (`6%` fraud vs `94%` legitimate), right-skewed/bounded `ClaimAmount` stratification, categorical fraud proportions (`ClaimType`, `ClaimSubmissionMethod`, `Cluster`), and numerical heatmaps. Note that EDA insights motivated explicit exclusion of raw identifier strings and `ClaimStatus`.

#### `scripts/preprocess.py` (`98` lines)
Executes four deterministic ETL stages:
1. **Calendar Parsing:** Converts `ClaimDate` via `pd.to_datetime` into `ClaimYear`, `ClaimMonth`, and `ClaimDayOfWeek` (`0=Monday .. 6=Sunday`).
2. **Column Purge:** Discards `ClaimID`, `PatientID`, `ProviderID`, `DiagnosisCode`, `ProcedureCode`, `ProviderLocation`, raw `ClaimDate`, and post-adjudication leaky target proxy `ClaimStatus`.
3. **Stratified Two-Step Split (`random_state=42`):** Partitions `(4500)` → `train (3150, 70%)` + `temp (1350, 30%)`, then splits `temp` symmetrically → `val (675, 15%)` + `test (675, 15%)`, preserving `~6%` positive class ratio in every partition (`189`, `41`, `40` frauds respectively).
4. **Train-Fitted Vectorization:** Fits `ColumnTransformer` with `StandardScaler()` on the `7` numeric columns and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on the `6` categorical columns (`22` output flags). Persists `models/preprocessor.joblib`, `data/processed/feature_names.txt`, and six `.npy` arrays.

#### `scripts/train_all_models.py` (`148` lines)
Orchestrates standardized training, timing, model selection, and artifact serialization:
1. Instantiates the five estimators (Section 4) inside a single configuration dictionary keyed by clean public names (`LogisticRegression`, `RandomForest`, `XGBoost`, `SVM`, `GradientBoosting`).
2. Fits each model on `(X_train, y_train)` while recording wall-clock fit seconds (`time.perf_counter()`).
3. Evaluates discrete predictions (`.predict(X)`) and continuous probability scores (`.predict_proba(X)[:, 1]`) across both `val` (`N=675`) and `test` (`N=675`).
4. Benchmarks batched inference latency across `100` claims (`inference_time_100_claims_sec`) and records exact serialized artifact sizes (`os.path.getsize(...)`).
5. Selects `best_model` strictly using maximum Validation F1 score (`Val_F1 = 1.0000` for `GradientBoosting`) and writes `evaluation/best_model.txt`, `evaluation/model_results.json`, and `logs/training_metrics.json`.

#### `scripts/comparison_plots.py` (`72` lines) & `scripts/error_analysis.py` (`55` lines)
`comparison_plots.py` consumes `model_results.json` to plot side-by-side ROC curves, Precision-Recall curves, bar charts comparing Accuracy/Precision/Recall/F1, fit latency comparisons, and heatmapped confusion matrices. `error_analysis.py` loads `best_model.txt` (`GradientBoosting.joblib`) and inspects individual test misclassifications (confirming `0` false positives and `1` false negative across `675` test records), writing concise textual summaries into `evaluation/result_analysis.md` and `result_analysis.txt`.

#### `scripts/create_notebooks.py` (`32` lines) — Documented Generator Bug
Programmatically constructs template `.ipynb` files for exploration. Inspection of lines `1–32` reveals an **uninterpolated format placeholder defect**: the Python code template references `{class_def}` inside a string block without supplying a corresponding `.format(class_def=...)` value or resolving it cleanly, meaning `create_notebooks.py` is broken as an automated generator and should not be assumed to produce runnable notebooks without template sanitization.

---

## 4. Deep-Dive Architectural & Mathematical Specification of All Five Estimators

The table below locks the exact constructor calls verified in `Approch-1/scripts/train_all_models.py` alongside serialized model keys and file sizes from `logs/training_metrics.json`:

| Dictionary Key / File Key | Actual Estimator Class in `train_all_models.py` | Verified Constructor Hyperparameters | Imbalance Handling Mechanism | Serialized File Size |
| :--- | :--- | :--- | :--- | :---: |
| **`LogisticRegression`** | `sklearn.linear_model.LogisticRegression` | `max_iter=2000`, `class_weight='balanced'`, `solver='lbfgs'`, `random_state=42` | Inverse class frequency weights ($w_0 \approx 0.532$, $w_1 \approx 8.333$) | `1,103 B` (`1.08 KB`) |
| **`RandomForest`** | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=300`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1` | Per-tree balanced class weight scaling in Gini impurity | `3,465,961 B` (`3.31 MB`) |
| **`XGBoost`** | `xgboost.XGBClassifier` | `use_label_encoder=False`, `eval_metric='logloss'`, `scale_pos_weight=2961/189`, `random_state=42`, `n_jobs=-1` | Positive gradient scale $p = 2961 / 189 \approx 15.6667$ | `85,788 B` (`83.78 KB`) |
| **`SVM`** | `sklearn.svm.SVC` | `probability=True`, `class_weight='balanced'`, `kernel='rbf'`, `random_state=42` | Asymmetric penalty $C_1 = C \cdot w_1$, $C_0 = C \cdot w_0$ | `96,459 B` (`94.20 KB`) |
| **`GradientBoosting`** | `sklearn.ensemble.HistGradientBoostingClassifier` | `random_state=42`, `class_weight='balanced'` | Histogram-binned boosting w/ balanced sample weights | `72,352 B` (`70.66 KB`) |

> **Important Naming Alert:** Although serialized under `models/GradientBoosting.joblib` and keyed `"GradientBoosting"`, the actual scikit-learn class instantiated on line 38 of `train_all_models.py` is **`HistGradientBoostingClassifier`**, NOT legacy `GradientBoostingClassifier`! All downstream consumers (`ml-service` and Track B comparison tables) inherit this historical key alias.

### 4.1 Model 1: `LogisticRegression` (L2-Regularized Linear Discriminator)

* **Decision Equation:** Learns parameter vector $\mathbf{w} \in \mathbb{R}^{29}$ and intercept $b \in \mathbb{R}$ predicting $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$ using L-BFGS quasi-Newton optimization capped at `2,000` iterations (`L2` penalty default `C=1.0`).
* **Geometric Behavior on Canonical Data:**
  Because fraudulent claims occupy the upper dollar interval `[\$7,465.07, \$9,978.43]` while `Cluster == 1` captures `98.89%` of frauds, a weighted linear combination places large positive weights on `Cluster` and `ClaimAmount`, shifting the decision plane cleanly below `\$7,465.07` when `Cluster == 1` to guarantee **`100.00%` Test Recall (`0` missed frauds)**.
* **Structural Limitation:** Because a single hyperplane $w^\top x + b = 0$ cannot bend orthogonally around complex interactions (e.g., excluding legitimate high-value claims sharing similar monetary and temporal ranges), `LogisticRegression` generates `24` false positives (`FPR = 3.78%`), depressing Test Precision to **`62.50%`** (`F1 = 76.92%`, `PR-AUC = 91.86%`).
* **Runtime Efficiency:** Lowest training footprint (`0.013s`), fastest 100-claim batch inference (`0.0003s` = `3 µs/claim`), and smallest serialized artifact (`1,103` bytes — just 30 floats + metadata).

### 4.2 Model 2: `RandomForest` (Bagged Orthogonal Decision Ensemble)

* **Algorithmic Structure:** Fits `300` independent unpruned CART decision trees over bootstrap replicates of the `3,150` training rows, sampling $\lfloor \sqrt{29} \rfloor = 5$ candidate features at each node split under class-balanced Gini impurity. Predicted probability averages binary leaf class votes weighted by sample balance across all `300` trees.
* **Empirical Performance Profile:**
  * Validation: `F1 = 98.77%`, `PR-AUC = 100.00%` (`TN=634, FP=0, FN=1, TP=40` — inferred from perfect Val PR & 1-error F1 matching).
  * Test (`N=675`): **`100.00%` Precision (`0` False Positives)**, `97.50%` Recall (`1` False Negative), `98.73%` F1, **`99.996%` ROC-AUC**, `99.94%` PR-AUC.
* **Operational Tradeoff:** Bagging `300` deep CART trees creates the **largest disk artifact (`3,465,961` bytes = `3.31 MB`, `~3,142×` larger than `LogisticRegression` and `~48×` larger than `HistGradientBoosting`)** and the slowest classical batch latency (`0.0537s` per 100 claims, `~179×` slower than `LogisticRegression` and `~90×` slower than `XGBoost`), yet yields unmatched false-positive resistance (`FP = 0`) and dominates Track B's `intake` (`ex-Cluster`) deployment audit!

### 4.3 Model 3: `XGBoost` (Gradient-Boosted Second-Order Trees)

* **Algorithmic Structure:** Sequentially adds `100` regularized regression trees (`xgb.XGBClassifier`) optimizing second-order Taylor approximations of balanced binary logistic loss where positive residuals are scaled by `scale_pos_weight = 2961 / 189 ≈ 15.6667`.
* **Peak Track A Test Leaderboard Winner:**
  Although Validation F1 (`98.77%`) tied `RandomForest` narrowly behind `HistGradientBoosting` (`100.00%`), on the held-out Test split (`N=675`), **`XGBoost` achieved the highest overall Track A test scores across three headline metrics**:
  * **Test PR-AUC:** **`100.00%`** (`1.0000`, highest of all models in Track A!)
  * **Test ROC-AUC:** **`100.00%`** (`1.0000`, perfect rank ordering of all 40 test frauds above all 635 legitimate claims!)
  * **Test F1:** **`98.77%`** (`TP=40, TN=634, FP=1, FN=0` — `100.00%` Recall, `97.56%` Precision).
* **Efficiency Balance:** Trains in **`0.028s`** (nearly `22×` faster than `RandomForest`), executes 100 claims in **`0.0006s`** (`90×` faster than `RandomForest`), and serializes cleanly in **`85,788` bytes (`83.78 KB`)**. Note why `Test PR-AUC = 100.00%` coexists with `FP = 1` (`F1 = 98.77%`): `XGBoost` ranked all `40` true frauds strictly higher in probability than every true negative (`FP = 1` occurred purely because one legitimate claim scored slightly above fixed threshold `0.50`, whereas the minimum true fraud scored even higher!).

### 4.4 Model 4: `SVM` (RBF Kernel Margin + Internal Platt Calibration)

* **Algorithmic Structure:** Solves the soft-margin quadratic dual optimization problem with Radial Basis Function kernel $K(x, x') = \exp(-\gamma \|x - x'\|_2^2)$ under class-balanced penalty multipliers (`class_weight='balanced'`), plus 5-fold cross-validated internal logistic Platt calibration enabled via `probability=True`.
* **CRITICAL DISCREPANCY ANALYSIS — `SVC.predict()` vs. `SVC.predict_proba() >= 0.5`:**
  Audits in both `Approch-1/evaluation/model_results.json` and `submission/results/legacy_artifact_audit.json` expose a textbook yet subtle scikit-learn trap inside `SVM.joblib`:
  * Calling `model.predict(X_test)` evaluates the **raw RBF SVM decision boundary** ($f_{\text{svc}}(x) \ge 0$), flagging **`51` positive predictions** (`TN=622, FP=13, FN=2, TP=38`), yielding **`Precision = 74.51%`, `Recall = 95.00%`, `F1 = 83.52%`**.
  * Calling `model.predict_proba(X_test)[:, 1] >= 0.5` evaluates the **5-fold Platt-calibrated sigmoid probability** $\sigma(A f_{\text{svc}}(x) + B) \ge 0.5$, flagging only **`39` positive predictions** (`TN=632, FP=3, FN=4, TP=36`), yielding **`Precision = 92.31%`, `Recall = 90.00%`, `F1 = 91.14%`**!
  * **Why This Happens:** Inside `sklearn.svm.SVC`, `predict()` ignores the fitted Platt parameters $(A, B)$ and uses the balanced margin hyperplane ($f(x) = 0$), whereas `predict_proba()` fits an unweighted logistic curve on internal fold decision scores! Because `ml-service/app/services/classical_service.py` uses `.predict_proba()` for live serving probability while `train_all_models.py` logged `.predict()` confusion matrices (`FP=13, FN=2`), offline `SVM` reports (`83.52%` F1) differ cleanly from `0.5`-thresholded `predict_proba` decisions (`91.14%` F1) documented in `legacy_artifact_audit.json`. Track B avoids this ambiguity entirely by instantiating `SVC(probability=False)` and applying analytical `scipy.special.expit(decision_function)` alongside validation-tuned thresholds.

### 4.5 Model 5: `GradientBoosting` (`HistGradientBoostingClassifier` — Track A Validation Winner)

* **Algorithmic Structure:** Discretizes continuous standardized features (`ClaimAmount`, `PatientIncome`, etc.) into integer histogram bins (inspired by LightGBM), growing histogram-based trees weighted inversely by training frequency (`class_weight='balanced'`, `random_state=42`).
* **Why Selected as Track A Best (`best_model.txt`):** On the `675`-sample validation set, `HistGradientBoostingClassifier` scored **`Val_F1 = 100.00%`** and **`Val PR-AUC = 100.00%`** (`0` errors on validation!), beating `RandomForest` and `XGBoost` (`98.77%` Val F1), and was persisted to `evaluation/best_model.txt` and analyzed in `scripts/error_analysis.py`.
* **Test Performance & Probability Histogram Quantization Artifact:** On Test (`N=675`), `GradientBoosting` achieves **`99.85%` Accuracy, `100.00%` Precision, `97.50%` Recall, `98.73%` F1 (`TN=635, FP=0, FN=1, TP=39`)** — matching `RandomForest`'s exact discrete test confusion counts while occupying just **`72,352` bytes** (notice prior docs `05_results.md` misprinted `723,352` — verified false by both `training_metrics.json` line 90 and NTFS metadata!). Notably, its Test `ROC-AUC = 98.47%` and `PR-AUC = 97.65%` sit slightly lower than `RandomForest` (`99.94%`) and `XGBoost` (`100.00%`) because its single missed fraud received a confident low histogram leaf score rather than hovering near threshold.

---

## 5. Complete Empirical Evaluation Benchmarks (`model_results.json` & `training_metrics.json`)

### 5.1 Master Validation (`N=675`, `41` Frauds) & Test (`N=675`, `40` Frauds) Performance Table

| Model Key | Val F1 | Val PR-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LogisticRegression`** | `84.21%` | `94.11%` | `96.44%` | `62.50%` | **`100.00%`** | `76.92%` | `99.50%` | `91.86%` |
| **`RandomForest`** | `98.77%` | **`100.00%`** | **`99.85%`** | **`100.00%`** | `97.50%` | `98.73%` | `99.996%` | `99.94%` |
| **`XGBoost`** | `98.77%` | `99.94%` | **`99.85%`** | `97.56%` | **`100.00%`** | **`98.77%`** | **`100.00%`** | **`100.00%`** |
| **`SVM`** *(`.predict`)* | `87.91%` | `94.42%` | `97.78%` | `74.51%` | `95.00%` | `83.52%` | `99.55%` | `92.52%` |
| **`SVM`** *(`proba>=0.5` audit)* | — | — | `98.96%` | `92.31%` | `90.00%` | `91.14%` | `99.55%` | `92.52%` |
| **`GradientBoosting`** *(HGB)* | **`100.00%`** | **`100.00%`** | **`99.85%`** | **`100.00%`** | `97.50%` | `98.73%` | `98.47%` | `97.65%` |

### 5.2 Complete Confusion-Matrix Counts & Derived Operating Diagnostics (Test Split, `N = 675`)

Every derived ratio below (Specificity, FPR, FNR, FDR, NPV, MCC) is computed algebraically from the exact observed `(TN, FP, FN, TP)` counts saved in `Approch-1/evaluation/model_results.json` (plus `legacy_artifact_audit.json` for calibrated `SVM`):

| Model | `TN` | `FP` | `FN` | `TP` | Total Errors | Specificity (TNR) | False Pos Rate (`FPR`) | False Neg Rate (`FNR`) | False Disc Rate (`FDR`) | Neg Pred Val (`NPV`) | Matthews `MCC` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LogisticRegression`** | `611` | `24` | **`0`** | **`40`** | `24` | `96.22%` | `3.78%` | **`0.00%`** | `37.50%` | **`100.00%`** | `0.7757` |
| **`RandomForest`** | **`635`** | **`0`** | `1` | `39` | **`1`** | **`100.00%`** | **`0.00%`** | `2.50%` | **`0.00%`** | `99.84%` | `0.9867` |
| **`XGBoost`** | `634` | `1` | **`0`** | **`40`** | **`1`** | `99.84%` | `0.16%` | **`0.00%`** | `2.44%` | **`100.00%`** | **`0.9870`** |
| **`SVM`** *(`.predict`)* | `622` | `13` | `2` | `38` | `15` | `97.95%` | `2.05%` | `5.00%` | `25.49%` | `99.68%` | `0.8304` |
| **`SVM`** *(`proba>=0.5`)* | `632` | `3` | `4` | `36` | `7` | `99.53%` | `0.47%` | `10.00%` | `7.69%` | `99.37%` | `0.9060` |
| **`GradientBoosting`** | **`635`** | **`0`** | `1` | `39` | **`1`** | **`100.00%`** | **`0.00%`** | `2.50%` | **`0.00%`** | `99.84%` | `0.9867` |

#### Surgical Takeaways from the Confusion-Matrix Audit

1. **Matthews Correlation Coefficient (`MCC`) Separates the Top Tier Cleanly:** Both `XGBoost` (`0.9870`) and `RandomForest` / `GradientBoosting` (`0.9867`) achieve near-unity correlation (`> 0.986`) despite `15.67:1` class imbalance, while `LogisticRegression` falls to `0.7757` purely from false-positive dilution (`24` alerts on benign high-value claims).
2. **`XGBoost` vs. `RandomForest`/`HGB` Cost Tradeoff:** On a fixed `0.50` threshold, `XGBoost` commits `1` false investigation (`FP=1, FN=0` -> zero leaked fraud payouts), whereas `RandomForest` and `GradientBoosting` commit `0` false investigations and miss `1` boundary fraud (`FP=0, FN=1`). Because `XGBoost` also achieved **`100.00%` Test PR-AUC**, sweeping its decision threshold marginally upward separates that lone `FP` cleanly from all `40` true positives!

### 5.3 Computational Efficiency, Latency & Storage Benchmark (`training_metrics.json`)

| Model Key | Fit Time (`s`) | Fit Speed vs. Fastest (`LR`) | Infer `100` Claims (`s`) | Avg Per-Claim Latency (`µs`) | Serialized Artifact Size (`Bytes`) | Footprint vs. `LR` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`LogisticRegression`** | **`0.013`** | `1.0x` | **`0.0003`** | **`3.0 µs`** | **`1,103`** | `1.0x` |
| **`XGBoost`** | `0.028` | `2.2x` slower | `0.0006` | `6.0 µs` | `85,788` | `77.8x` |
| **`GradientBoosting`** | `0.192` | `14.8x` slower | `0.0013` | `13.0 µs` | `72,352` | `65.6x` |
| **`SVM`** | `0.285` | `21.9x` slower | `0.0025` | `25.0 µs` | `96,459` | `87.5x` |
| **`RandomForest`** | `0.625` | `48.1x` slower | `0.0537` | `537.0 µs` | `3,465,961` | `3,142.3x` |

**Efficiency Pareto Winner (`HistGradientBoostingClassifier` & `XGBClassifier`):** `XGBoost` delivers **`100.00%` Test PR-AUC** at `6 µs/claim` in `85.8 KB`, while `HistGradientBoosting` delivers `100%` Test Precision in only `72.4 KB` (`48x` smaller and `41x` faster at batched scoring than `RandomForest`).

---

## 6. Live Serving Integration Path (`ml-service/`)

When a claim hits `POST /predict/ml` in FastAPI (`ml-service/app/services/classical_service.py` + `app/api/predictions.py`):

1. `ClassicalService` eagerly loads `Approch-1/models/preprocessor.joblib`, reads `Approch-1/data/processed/feature_names.txt`, and caches all five `.joblib` estimators into an in-memory registry keyed identically by `LogisticRegression`, `RandomForest`, `XGBoost`, `SVM`, and `GradientBoosting`.
2. Raw JSON payloads matching Pydantic `ClaimRequest` (`extra="forbid"`) extract `ClaimYear`, `ClaimMonth`, `ClaimDayOfWeek` from `ClaimDate`, build a single-row pandas `DataFrame` matching the training column sequence, invoke `preprocessor.transform(df)`, and pass the resulting `(1, 29)` matrix to the requested estimator's `.predict_proba()` method.
3. Because `ml-service` evaluates `.predict_proba()[:, 1] >= 0.5` cleanly across every model (preserving scikit-learn probability outputs uniformly), callers invoking `"SVM"` see Platt-calibrated decisions (`39` flagged / `91.14%` F1 behavior) rather than raw `.predict()` decisions (`51` flagged / `83.52%` F1).

---

## 7. Operational Runbook, Reproducibility Caveats & Verified Errata

### 7.1 Step-by-Step Re-Execution Commands (Windows PowerShell)

```powershell
# 1. Enter Approach 1 workspace
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\Approch-1

# 2. Create required Windows NTFS junction for legacy 'project/' relative paths (one-time)
if (-not (Test-Path .\project)) {
    New-Item -ItemType Junction -Path ".\project" -Target "."
}

# 3. Activate Python 3.11 virtual environment & install dependencies if needed
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 4. Run full end-to-end Classical ML pipeline
python scripts/eda_plots.py
python scripts/preprocess.py
python scripts/train_all_models.py
python scripts/comparison_plots.py
python scripts/error_analysis.py
```

### 7.2 Verified Errata & Documentation Discrepancies Inside `Approch-1/`

| Item / Artifact | Observed Fact in Code / Logs | Erroneous Claim / Hazard Elsewhere | Impact Assessment |
| :--- | :--- | :--- | :--- |
| **`GradientBoosting.joblib` Estimator Type** | Instantiates `HistGradientBoostingClassifier` (`train_all_models.py` L38). | Filename suggests vanilla `GradientBoostingClassifier`. | Critical for parameter parity in Track B (`max_iter=100`, binned trees). |
| **`GradientBoosting.joblib` Byte Size** | Verified **`72,352` bytes** (`training_metrics.json` L90 & disk stats). | [`documentation/05_results.md`](../../Approch-1/documentation/05_results.md) printed `723,352` bytes (`10x` typo). | Reconciled cleanly in Vol. 08 Ledger. |
| **`SVM` Dual Evaluation Semantics** | `.predict()` flags `51` (`F1=83.52%`), `.predict_proba()>=0.5` flags `39` (`F1=91.14%`). | Mixing raw `.predict` confusion counts beside Platt `predict_proba` AUC without flagging calibration gap. | Track B replaces `probability=True` with `expit(decision_function)`. |
| **`scripts/create_notebooks.py`** | Contains unpopulated `{class_def}` string interpolation variable (`lines 1–32`). | Assumed in folder README listings to build runnable Jupyter notebooks. | Do not run as part of automated CI; notebooks must be generated manually or fixed in template. |
| **`Cluster` Inclusion (`29f`)** | Includes post-hoc `Cluster` feature (`idx 6`), driving near-ceiling validation/test metrics. | Can mislead reviewers into expecting `~99.9%` F1 without verifying `Cluster` availability at claim intake. | Audited directly via `intake` (`ex-Cluster`) & `temporal` Track B runs in Volume 07. |
