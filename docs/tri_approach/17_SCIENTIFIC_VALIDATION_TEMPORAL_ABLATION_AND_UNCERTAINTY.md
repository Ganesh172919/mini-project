# Volume 17 — Scientific Validity, Temporal Drift, Ablation Proofs & Uncertainty Quantification

> **Document Classification:** Experimental Methodology, Threats-to-Validity & Statistical Rigor Monograph  
> **Scope:** Rigorous causal & empirical examination of why models succeed or fail across `baseline`, `intake`, `no_amount`, `temporal`, `Autoencoder`, and `B=1000` bootstrap distributions.  
> **Primary Ground-Truth Sources:** [`submission/results/dataset_audit.json`](../../submission/results/dataset_audit.json), [`submission/results/selection.json`](../../submission/results/selection.json), [`submission/results/analysis.json`](../../submission/results/analysis.json), [`submission/results/bootstrap.json`](../../submission/results/bootstrap.json), [`submission/results/metrics.csv`](../../submission/results/metrics.csv), [`Approch-1/evaluation/model_results.json`](../../Approch-1/evaluation/model_results.json), and [`Approch-2-DL/results/*_metrics.json`](../../Approch-2-DL/results/autoencoder_metrics.json).

---

## 1. Scientific Skepticism Checklist: Beyond `99.85%` Headline Benchmarks

Reporting `99.85%` Accuracy or `100.00%` PR-AUC on a `4,500`-row tabular dataset without controlled ablation invites severe deployment failures caused by silent dataset artifacts. The dual-track architecture in this repository tests four fundamental scientific validity hypotheses systematically:

| Hypothesis Under Test | Naive Experimental Illusion | Controlled Intervention Executed | Verified Empirical Verdict |
| :--- | :--- | :--- | :--- |
| **`H1: Post-Hoc Clustering Leakage`** | "`XGBoost` & `HGB` solve fraud almost effortlessly (`100%` Val F1)." | Drop `Cluster` (`98.89%` frauds in Cluster `1`) in Track B **`intake`** (`28` features). | `RandomForest` retains `1.0000` Test AP (`0.9873` F1, `thr=0.4167`), proving genuine patterns persist *even without* `Cluster`. |
| **`H2: Demographic & Structural Signal Depth`** | "Demographic, specialty, and channel metadata carry broad predictive signal." | Drop `ClaimAmount` in Track B **`no_amount`** (`run_pipeline.py`). | **Refuted:** Test AP crashes across *every* model (`0.1869` `SVM` to `0.2404` `GB`, `0.2066` `RF`). |
| **`H3: Temporal Extrapolation Invariance`** | "Deep Learning (`CNN1D`, `ResMLP`, `MLP`) matches trees identically in deployment." | Enforce strict chronological split (`>= 2024-03-11`) in **`temporal`** (`N_test=671`). | **Tree Superiority Proven:** Trees retain `1.0000` AP / `1.0000` F1, while `CNN1D` (`0.9172`), `MLP` (`0.8581`), & `ResMLP` (`0.7979`) decay cleanly. |
| **`H4: Benign Manifold Anomaly Assumption`** | "Unsupervised `Autoencoder` detects upper-tail fraud via reconstruction error." | Train `29→32→10→32→29` AE on benign rows (`y==0`, `N=2,961`), evaluate Test. | **Refuted:** `46.02%` ROC-AUC, `5.23%` PR-AUC (`TP=1, FN=39`, `MCC=-0.0192`). |

---

## 2. Quantitative Leakage Dissection (`Cluster`, `ClaimStatus`, `ClaimAmount`)

### 2.1 Why `Cluster` Breaks Live Pre-Adjudication Fairness (`dataset_audit.json`)

| Segment | Legitimate (`y=0`) | Fraud (`y=1`) | Intra-Cluster Rate | Conditional Entropy & Deployment Implication |
| :--- | :---: | :---: | :---: | :--- |
| `Cluster 0` | `1,144` | `0` | `0.00%` | Zero frauds; a split rule `Cluster != 1` removes nearly `75%` of negatives (`Cluster 0 + 2 + ~3`). |
| `Cluster 1` | `833` | `267` | **`24.27%`** | Holds **`98.89%` (`267 / 270`)** of all frauds in the corpus! |
| `Cluster 2` | `1,104` | `0` | `0.00%` | Zero frauds. |
| `Cluster 3` | `1,149` | `3` | `0.26%` | Only `3` edge-case frauds outside Cluster `1`. |

Because batch unsupervised clustering relies on global corpus density estimation (including `ClaimAmount` correlations computed across the entire spreadsheet), supplying `Cluster` during live scoring leaks batch structure not available at point-of-care EDI claim submission (`ADR-07`). Requiring Track B winner selection (`selection.json`) exclusively over the **`intake` (`ex-Cluster`) scenario** provides an authentic production baseline.

### 2.2 Geometry of Monetary Conditioning & the `no_amount` Collapse

Recall the empirical bounds verified in `dataset_audit.json`:

* Fraudulent `ClaimAmount`: `min = \$7,465.07`, `max = \$9,978.43`, `mean = \$8,733.5651`
* Legitimate `ClaimAmount`: `min = \$100.12`, `max = \$9,997.20`, `mean = \$4,776.7978`

Why does removing `ClaimAmount` collapse Test AP from `~1.000` to `0.1869–0.2404`?

1. **Independence of Non-Monetary Marginals:** Patient demographics (`Age`, `Gender`, `MaritalStatus`, `EmploymentStatus`), `ProviderSpecialty`, `ClaimType`, and `ClaimSubmissionMethod` overlap heavily between legitimate and fraudulent rows across `\$100–\$9,997`.
2. **Residual `no_amount` Lift Driven by `Cluster 1` Baseline:** In `no_amount` (`submission/scripts/run_pipeline.py`), a model seeing `Cluster == 1` predicts base cluster risk (`24.27%` prior inside `Cluster 1`), explaining why Average Precision lands right around **`0.1869–0.2404`** (`~0.24` matches `267 / 1100 = 0.2427`!). Without both `ClaimAmount` and `Cluster`, non-monetary tabular features lack separation in this synthetic corpus.

---

## 3. Out-of-Time Chronological Generalization Mechanics (`temporal` Split)

### 3.1 Partition Boundaries & Class Prevalence Drift

Track B splits chronologically along `ClaimDate` (`731` dates spanning `2022-07-09` through `2024-07-08`) at cutoffs `["2023-11-21", "2024-03-10"]`:

* **Train (`2022-07-09 .. 2023-11-20`):** `3,152` records (`199` frauds, `2,953` legit, **`6.313%` prevalence**)
* **Validation (`2023-11-21 .. 2024-03-10`):** `677` records (`34` frauds, `643` legit, **`5.022%` prevalence**)
* **Test (`2024-03-11 .. 2024-07-08`):** `671` records (`37` frauds, `634` legit, **`5.514%` prevalence**)

Notice both **concept prior drift** (`6.31% → 5.02% → 5.51%`) and **covariate coordinate extrapolation** in standardized `ClaimYear` (`idx 3`)!

### 3.2 Mathematical Explanation for the Tree vs. Neural Divergence

Compare out-of-time Test scores on `temporal` (`N_test = 671`, `37` frauds):

| Classifier Family | Observed `AP` | Observed `F1` | Geometric Behavior When `ClaimYear` Shifts to `2024` |
| :--- | :---: | :---: | :--- |
| **`RandomForest` / `XGBoost` / `HGB`** | **`1.0000`** | **`1.0000`** | Decision trees greedily select high-gain `ClaimAmount` & `Cluster` splits at root depth; irrelevant `ClaimYear` branches rarely appear, keeping test leaves completely invariant to future years! |
| **`CNN1D`** | `0.9172` | `0.8108` | `Conv1d(k=3)` local receptive windows `[0..2]` isolate `(ClaimAmount, PatientAge, PatientIncome)` cleanly before seeing `[3..5]` calendar features, buffering drift compared to global dense MLPs. |
| **`LogisticRegression`** | `0.8799` | `0.7397` | Additive term `w_year * z_year` linearly shifts log-odds intercepts monotonically across unseen 2024 timestamps. |
| **`MLP`** | `0.8581` | `0.6957` | Dense `Linear(29, 64)` projections combine non-zero random/fitted `ClaimYear` weights into ReLU hidden gates across the entire input layer. |
| **`SVM` (`RBF + expit`)** | `0.8396` | `0.7619` | Radial similarity $\exp(-\gamma \|x - x_{sv}\|^2)$ decays exponentially whenever $(z_{\text{year, test}} - z_{\text{year, sv}})^2$ grows out-of-domain! |
| **`ResidualMLP`** | **`0.7979`** | **`0.7229`** | Identity skip additions preserve `Linear(29, 64)` calendar coordinate shift straight through both residual blocks, amplifying continuous drift errors (`-20.21` AP pts vs trees!). |

**Practical Scientific Recommendation:** Any neural or kernel classifier deployed onto longitudinal healthcare billing streams should exclude monotonically increasing absolute timestamps (`ClaimYear`) from raw continuous normalization, encoding strictly seasonal cyclical features (`sin/cos(Month)`, `DayOfWeek`) or time-relative velocity ratios instead.

---

## 4. Bootstrap Uncertainty Quantification (`bootstrap.json`, `B = 1,000`)

With `40` positive test cases (`N = 675`), discrete accuracy counts exhibit quantized granularity (`1` misclassified positive equals `1/40 = 2.50%` recall swing!). Track B's `bootstrap.json` (`B = 1,000`, `seed = 2026`) quantifies sampling spread for the selected `intake` `RandomForest` model:

* **`AP` (`[1.0, 1.0]`):** Zero uncertainty width (`0.0000`) because **every single test fraud (including `Row 1124` at `score = 0.27`) scored strictly above all `635` legitimate test records**, so rank ordering remains `1.0000` regardless of which claim subset bootstrap resampling duplicates!
* **`F1` (`[0.9610, 1.0000]`):** Reflects finite sampling variation in threshold crossing around `0.4167`: resamples excluding `Row 1124` achieve `1.0000` F1, whereas resamples drawing multiple copies of boundary `Row 1124` pull F1 down toward `0.9610` (`2–3` duplicated `FN` hits).
* **`Recall` (`[0.9250, 1.0000]`):** Tight upper bound (`>= 92.5%` recall at `95%` confidence even without `Cluster`).

---

## 5. Comprehensive Threats-to-Validity Registry

| Validity Category | Identified Limitation in Current Corpus | Impact on Generalizability | Mitigation Inside Repository / Future Work |
| :--- | :--- | :--- | :--- |
| **External Validity (Dataset Synthetic Scale)** | `4,500` claims with `4,500` unique `PatientID`/`ProviderID` keys (`1` claim per provider/patient) and zero frauds below `\$7,465.07`. | Real insurer clearinghouses feature repeat organized fraud rings (`1:N` patient/provider graphs) and low-dollar unbundling schemes (`< \$1,000`). | Document explicitly that `no_amount` & `temporal` stress tests isolate boundary physics on this workbook, not multi-provider graph fraud. |
| **Construct Validity (`AgenticEnsemble-v1`)** | Active `:3000`/`8020` agentic rule saturates at `\$260–\$300` while claiming `model_name="AgenticEnsemble-v1"`. | Presenting Agentic UI metrics without clarifying its rule-stub nature conflates UI choreography tests with statistical accuracy. | Explicitly quarantined in `Vol. 05` & `Vol. 11` Cards + tracked for Tier 1 wiring. |
| **Internal Validity (`SVC` Dual Outputs)** | `SVM.joblib` emits diverging F1 between `.predict()` (`83.52%`) and `.predict_proba()` (`91.14%`). | Mixing raw hyperplane confusion counts with Platt AUC distorts model comparisons. | Resolved in `legacy_artifact_audit.json` & replaced with `probability=False + expit` in Track B. |
| **Conclusion Validity (`Autoencoder`)** | Unsupervised 95th percentile cutoff evaluated on overlapping `\$7.4k–\$10k` amounts yields `5.23%` PR-AUC. | Demonstrates clearly when one-class L2 reconstruction should *not* be chosen. | Preserved intentionally as empirical negative evidence (`Vol. 04`, `Vol. 10`). |
