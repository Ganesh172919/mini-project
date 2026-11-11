# Volume 02 — Mathematical Formulation, Forensic Data Audit & Feature Contracts

> **Document Classification:** Mathematical Specification, Dataset Forensics & Schema Contract  
> **Canonical Dataset:** `Health Insurance Fraud Claims.xlsx` (`SHA-256: 952b6e23c9845b9086994c08bc81323575968c318f845ea69e3b9609614b6a45`)  
> **Primary Ground-Truth Sources:** [`Approch-1/data/processed/feature_names.txt`](../../Approch-1/data/processed/feature_names.txt), [`Approch-1/scripts/preprocess.py`](../../Approch-1/scripts/preprocess.py), [`Approch-2-DL/processed/contract_metadata.json`](../../Approch-2-DL/processed/contract_metadata.json), [`submission/results/dataset_audit.json`](../../submission/results/dataset_audit.json), [`submission/models/intake/contract.json`](../../submission/models/intake/contract.json), [`ml-service/app/schemas/claims.py`](../../ml-service/app/schemas/claims.py), and [`docs/02_PROBLEM_FORMULATION.md`](../02_PROBLEM_FORMULATION.md).

---

## 1. Formal Decision-Theoretic & Optimization Problem Statement

Automated healthcare claim fraud detection across Approaches 1 and 2 operates as a supervised (and baseline unsupervised) tabular mapping from raw mixed-type insurance claim records into calibrated or uncalibrated risk scores and discrete adjudication triggers.

### 1.1 Domain Spaces & Feature Projection Pipeline

Let the raw input claim record belong to a heterogeneous structured domain $\mathcal{X}_{\text{raw}}$ spanning `19` tabular columns documented in `Health Insurance Fraud Claims.xlsx`. After identifier pruning (`6` high-cardinality keys), outcome leakage removal (`ClaimStatus`), and policy-conditional inclusion of `Cluster` and `ClaimAmount`, an explicit feature transformation map $\Phi_{\mathcal{P}}$ projects each valid record into a fixed Euclidean vector space:

$$\Phi_{\mathcal{P}} : \mathcal{X}_{\text{raw}} \longrightarrow \mathbb{R}^{d_{\mathcal{P}}}$$

where the output dimensionality $d_{\mathcal{P}}$ depends on the governing evaluation policy $\mathcal{P}$:

| Policy Profile ($\mathcal{P}$) | Track / Directory | Numeric Features ($d_{\text{num}}$) | OHE Binary Columns ($d_{\text{cat}}$) | Total Width ($d_{\mathcal{P}}$) | Inclusion of `Cluster` & `ClaimAmount` |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Canonical Track A / `baseline`** | `Approch-1/`, `Approch-2-DL/`, `ml-service/`, `submission/` | `7` | `22` | **`29`** | Includes both `Cluster` and `ClaimAmount` |
| **Deployment `intake`** | `submission/models/intake/` (`ADR-07`) | `6` | `22` | **`28`** | Drops `Cluster`; retains `ClaimAmount` |
| **Sensitivity `no_amount`** | `submission/` ablation (`run_pipeline.py`) | Drops `ClaimAmount` (`6` if `Cluster` kept, or ex-amount numeric subset) | `22` | `27` or `28` *(see Track B)* | Drops `ClaimAmount` to isolate monetary dependence |
| **Out-of-Time `temporal`** | `submission/` chronological holdout | `7` or intake-aligned temporal split | `22` | `29` / policy-aligned | Tests forward time drift from `2024-03-11` onward |

Each transformed training dataset constitutes an empirical paired sample $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^{N_{\text{train}}}$ where $x_i = \Phi_{\mathcal{P}}(\tilde{x}_i) \in \mathbb{R}^{d}$ and $y_i \in \{0, 1\}$ denotes ground-truth claim legitimacy (`0 = Legitimate`, `1 = Fraud`). Across the full corpus (`N = 4,500`), positive prevalence is:

$$\pi = \frac{1}{N}\sum_{i=1}^{N} y_i = \frac{270}{4500} = 0.0600 \quad (6.00\%)$$

with $N_{\text{neg}} = 4,230$ legitimate records ($94.00\%$) and raw class-imbalance odds ratio:

$$r = \frac{N_{\text{neg}}}{N_{\text{pos}}} = \frac{4230}{270} = \frac{47}{3} \approx 15.6667$$

### 1.2 Cost-Sensitive Empirical Risk Minimization & Exact Reweighting Constants

Because missing an actual fraudulent claim ($FN$) incurs direct financial payout loss while misclassifying a legitimate claim ($FP$) incurs manual SIU audit overhead, standard unweighted $0-1$ empirical risk minimization collapses under $\pi = 0.06$. Instead, every supervised learner across Approaches 1 and 2 optimizes an asymmetric weighted surrogate loss:

$$\min_{\theta} \; \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \left[ w_1 \, y_i \, \ell\big(f_\theta(x_i), 1\big) + w_0 \, (1 - y_i) \, \ell\big(f_\theta(x_i), 0\big) \right] + \Omega(\theta)$$

Two mathematically isomorphic parameterizations appear directly in source code:

1. **Balanced Inverse-Frequency Sample Weights (`class_weight='balanced'` in `scikit-learn`):**
   Used by `LogisticRegression`, `RandomForestClassifier`, `SVC`, and `HistGradientBoostingClassifier` inside [`Approch-1/scripts/train_all_models.py`](../../Approch-1/scripts/train_all_models.py) and `submission/scripts/run_pipeline.py`:
   $$w_c = \frac{N_{\text{train}}}{2 \, N_{\text{train}, c}}, \qquad c \in \{0, 1\}$$
   Evaluating explicitly on the Track A random training split ($N_{\text{train}} = 3,150$, $N_{\text{neg}} = 2,961$, $N_{\text{pos}} = 189$):
   $$w_0 = \frac{3150}{2 \times 2961} \approx 0.531915, \qquad w_1 = \frac{3150}{2 \times 189} = \frac{25}{3} \approx 8.333333$$
   The minority-to-majority gradient amplification factor is identically:
   $$\frac{w_1}{w_0} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = \frac{2961}{189} = \frac{47}{3} \approx 15.666667$$

2. **Minority Multiplier Scaling (`scale_pos_weight` in `XGBoost` & `pos_weight` in `PyTorch`):**
   Both `XGBClassifier` (`scale_pos_weight = neg / pos`) and PyTorch's `nn.BCEWithLogitsLoss(pos_weight=...)` normalize majority sample weight to $w_0 = 1$ and scale strictly positive samples by:
   * **Random Stratified Training Split ($N_{\text{train}} = 3,150$):** $p = \frac{2961}{189} = 15.666667$
   * **Chronological Temporal Training Split ($N_{\text{train}} = 3,152$, $N_{\text{pos}} = 199$, $N_{\text{neg}} = 2,953$):**
     $$p_{\text{temp}} = \frac{2953}{199} \approx 14.839196$$

### 1.3 Weighted Binary Cross-Entropy & Posterior Log-Odds Distortion Proof

Consider all four supervised PyTorch architectures (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`), each terminating in an unconstrained scalar logit $z_i = f_\theta(x_i) \in \mathbb{R}$. Training minimizes weighted Binary Cross-Entropy with Logits:

$$\mathcal{L}_{\text{wBCE}}(\theta) = -\frac{1}{B} \sum_{i=1}^{B} \Big[ p \cdot y_i \log \sigma(z_i) + (1 - y_i) \log\big(1 - \sigma(z_i)\big) \Big], \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

#### Exact Gradient & Logit Shift Derivation

Differentiating the sample term $\ell_i(z_i, y_i)$ with respect to pre-activation logit $z_i$ via $\frac{d}{dz}\log \sigma(z) = 1 - \sigma(z)$ and $\frac{d}{dz}\log(1-\sigma(z)) = -\sigma(z)$ yields:

$$\frac{\partial \ell_i}{\partial z_i} = -p \cdot y_i \big(1 - \sigma(z_i)\big) + (1 - y_i)\sigma(z_i) = \sigma(z_i)\big(1 + (p - 1)y_i\big) - p \cdot y_i$$

Inspecting both ground-truth classes highlights the structural asymmetry:

* For a legitimate example ($y_i = 0$): $\frac{\partial \ell_i}{\partial z_i} = \sigma(z_i) \in (0, 1)$.
* For a fraudulent example ($y_i = 1$): $\frac{\partial \ell_i}{\partial z_i} = p\big(\sigma(z_i) - 1\big) \in (-p, 0)$, scaling downward updates on logits by factor $p \approx 15.67\times$.

Now suppose a sufficiently expressive model reaches Bayes optimality pointwise given true conditional fraud probability $\eta(x) = \mathbb{P}(Y = 1 \mid X = x)$. Setting expected pointwise derivative to zero:

$$\mathbb{E}\left[\frac{\partial \ell}{\partial z} \;\middle|\; X = x\right] = \eta(x) \cdot p\big(\sigma(z^*(x)) - 1\big) + \big(1 - \eta(x)\big)\sigma(z^*(x)) = 0$$

Solving explicitly for $\sigma(z^*(x))$:

$$\sigma(z^*(x)) = \frac{p \cdot \eta(x)}{p \cdot \eta(x) + 1 - \eta(x)} \implies z^*(x) = \log\left(\frac{\eta(x)}{1 - \eta(x)}\right) + \log(p)$$

Substituting $p = 2961 / 189 \approx 15.6667$:

$$\log(p) = \ln(15.6667) \approx +2.7515$$

#### Engineering Consequence (`ADR-05`: `score_is_calibrated = False`)

This exact analytical result proves **why scores emitted by pos-weighted classifiers cannot be interpreted as calibrated fraud probabilities without temperature/Platt recalibration or prior correction**:

* A claim whose true empirical posterior fraud likelihood is just $\eta(x) = 0.06$ (population average!) receives an optimal pos-weighted probability of:
  $$\sigma(z^*(x)) = \frac{15.6667 \times 0.06}{15.6667 \times 0.06 + 0.94} = \frac{0.94}{0.94 + 0.94} = 0.5000$$
* Consequently, a raw threshold of $0.50$ on a `pos_weight`-trained neural network or `class_weight='balanced'` classifier corresponds mathematically to classifying at true posterior probability $\eta(x) \ge \pi = 0.06$, deliberately sacrificing precision unless the threshold is swept higher on validation data (exact motivation for Track B's validation F1 sweep `selection.json`).

### 1.4 Unsupervised Autoencoder Formulation & Anomaly Calibration Map

In [`Approch-2-DL/scripts/dl_advanced_models.py`](../../Approch-2-DL/scripts/dl_advanced_models.py) and `train_advanced.py`, the symmetric `Autoencoder` trains exclusively on normalized legitimate training records $\mathcal{D}_{\text{train}, 0} = \{x_i \in \mathcal{D}_{\text{train}} : y_i = 0\}$ ($N = 2,961$):

$$\hat{x}_i = D_\psi\big(E_\phi(x_i)\big), \qquad E_\phi : \mathbb{R}^{29} \to \mathbb{R}^{10}, \quad D_\psi : \mathbb{R}^{10} \to \mathbb{R}^{29}$$

$$\min_{\phi, \psi} \; \frac{1}{|\mathcal{D}_{\text{train}, 0}|} \sum_{x_i \in \mathcal{D}_{\text{train}, 0}} \frac{1}{29} \|x_i - \hat{x}_i\|_2^2$$

Notice that `Autoencoder` performs **no explicit weight penalty on fraud**, nor does it receive negative supervision. Per-claim anomaly scoring computes scalar Mean Squared Error reconstruction error:

$$e(x) = \frac{1}{29} \sum_{j=1}^{29} \big(x_j - \hat{x}_j\big)^2$$

The binary anomaly cutoff $\tau$ is calibrated non-parametrically on the full validation split ($N_{\text{val}} = 675$) at the 95th percentile (`np.percentile(..., 95)`), recorded verbatim in `Approch-2-DL/results/autoencoder_metrics.json`:

$$\tau_{0.95} = 0.12858837842941284$$

When serving the `Autoencoder` via FastAPI (`ml-service/app/services/dl_service.py`), raw unbounded $e(x) \in [0, \infty)$ is mapped smoothly into $(0, 1)$ via a centered logistic sigmoid scaled by factor $5 / \tau$:

$$s_{\text{AE}}(x) = \sigma\!\left(\frac{5 \cdot \big(e(x) - \tau\big)}{\max(\tau, 10^{-8})}\right)$$

Evaluating this monotonic transform at operational landmarks demonstrates its strict threshold preservation (`is_fraud = (e(x) >= tau)` coincides with `s_AE >= 0.5`):

| Reconstruction Error Regime | Normalized Ratio $e(x) / \tau$ | Logit Argument | Served `dl_service.py` Score $s_{\text{AE}}(x)$ | Classification |
| :--- | :---: | :---: | :---: | :--- |
| Zero error ($e = 0$) | $0.00$ | $-5.00$ | $0.006693$ (`0.67%`) | Legitimate |
| Half threshold ($e = 0.5\tau \approx 0.0643$) | $0.50$ | $-2.50$ | $0.075858$ (`7.59%`) | Legitimate |
| Exact cutoff ($e = \tau \approx 0.1286$) | $1.00$ | $0.00$ | $0.500000$ (`50.00%`) | Flagged Boundary |
| Moderate anomaly ($e = 1.5\tau \approx 0.1929$) | $1.50$ | $+2.50$ | $0.924142$ (`92.41%`) | Flagged Fraud |
| Severe error ($e = 2.0\tau \approx 0.2572$) | $2.00$ | $+5.00$ | $0.993307$ (`99.33%`) | Flagged Fraud |

Notice again why `score_is_calibrated = False`: $s_{\text{AE}}(x) = 0.924$ represents reconstruction error $50\%$ above $\tau_{0.95}$, **not** a $92.4\%$ probability of fraud!

### 1.5 Approach 3 Heuristic Scoring Formula (`AgenticEnsemble-v1`)

Both `Approach-3-Agentic-AI/app/api/predict.py` and `backend/node-api/src/agentic/predict.ts` compute a stateless continuous risk scalar $s_{\text{agentic}} \in [0, 1]$ directly from two raw inputs (`claim_amount`, `patient_age`):

$$s_{\text{agentic}}(\text{amt}, \text{age}) = \operatorname{clip}\!\Big(0.01 \cdot \text{amt} + 0.005 \cdot \text{age} - 2.0,\; 0.0,\; 1.0\Big)$$

Its saturation regime (`claim_amount >= 250 - 0.5 * patient_age` yields `1.0`) is analyzed rigorously in Volume 05.

---

## 2. Evaluation Metrics Theory Under Extreme Class Skew

To avoid false security from majority-class accuracy, all experiments report six core scalar metrics computed directly from confusion-matrix quadruples $(TN, FP, FN, TP)$ and ranked continuous prediction scores $s_i$:

### 2.1 Closed-Form Metric Definitions

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \qquad \text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall (TPR)} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,TP}{2\,TP + FP + FN}$$

For deeper confusion-matrix forensics in Volumes `03`, `04`, and `07`, we also compute derived operating statistics cleanly from observed $(TN, FP, FN, TP)$ integers:

$$\text{FPR} = \frac{FP}{FP + TN}, \qquad \text{Specificity (TNR)} = \frac{TN}{TN + FP}, \qquad \text{NPV} = \frac{TN}{TN + FN}$$

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### 2.2 Average Precision (`PR-AUC`) vs. `ROC-AUC` on `N_test = 675`

Let predicted scores $\{s_i\}$ be sorted descending so threshold sweeps $\tau_n$ trace precision-recall pairs $(R_n, P_n)$. Both `scikit-learn` (`average_precision_score`) and Track B compute step-interpolated Average Precision:

$$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$

Crucially, contrast random-guessing baselines on the test partition ($N = 675$, $N_{\text{pos}} = 40$, $N_{\text{neg}} = 635$):

* **Random ROC-AUC Baseline:** Always $0.5000$ regardless of skew.
* **Random PR-AUC Baseline:** Equals test prevalence $\pi_{\text{test}} = \frac{40}{675} \approx 0.05926$ (`5.93%`).

#### Quantitative Proof Using Actual Approach 1 Test Counts

Consider Approach 1's `LogisticRegression` test confusion matrix: `TN = 611`, `FP = 24`, `FN = 0`, `TP = 40` (`Precision = 62.50%`, `Recall = 100.00%`):

* **ROC False Positive Rate:** $FPR = \frac{24}{24 + 611} = \frac{24}{635} = 3.7795\%$. With `TPR = 100%` and only `3.78%` FPR, ROC-AUC remains compressed near ceiling at **`99.50%`**.
* **PR Precision Cost:** Adding `24` false alarms alongside `40` true frauds forces `24 / 64 = 37.50%` of investigator audits to be wasted on legitimate claimants, dragging Precision down to `62.50%` and separating **PR-AUC (`91.86%`)** sharply from tree ensembles (`99.94%–100.00%`).
* **Autoencoder Random-Equivalent Check:** Approach 2's `Autoencoder` records `46.02%` ROC-AUC (slightly worse than coin flip `50%`) and `5.23%` PR-AUC (slightly worse than random prevalence `5.93%`). Only PR-AUC immediately signals that unsupervised MSE ranking carries zero positive precision lift over random sampling.

### 2.3 Validation-Only Threshold Optimization (`ADR-12`) & Bootstrap CIs

While Track A models freeze cutoff $\tau = 0.50$ (or `Autoencoder` 95th validation percentile), Track B eliminates arbitrary cutoff mismatch under cost-sensitive weighting via strict validation F1 selection:

$$\tau^* = \arg\max_{\tau \in \{s_i : i \in \mathcal{D}_{\text{val}}\}} F_1\big(\{y_i, \mathbb{I}[s_i \ge \tau]\}_{i \in \mathcal{D}_{\text{val}}}\big)$$

Uncertainty around point metrics on the `675`-sample test split is then quantified via non-parametric stratified/random bootstrap resampling ($B = 1,000$ resamples, `seed = 2026` in `submission/results/bootstrap.json`), reporting 2.5th and 97.5th quantiles $[q_{0.025}, q_{0.975}]$.

---

## 3. Forensic Audit of `Health Insurance Fraud Claims.xlsx`

Every experiment across Track A (`Approch-1/`, `Approch-2-DL/`) and Track B (`submission/`) originates from a single immutable Excel workbook audited programmatically by `submission/scripts/common.py` and recorded in `submission/results/dataset_audit.json`.

### 3.1 Workbook Provenance & Integrity Profile

| Audit Attribute | Verified Observed Value (`dataset_audit.json`) | Significance |
| :--- | :--- | :--- |
| **Canonical File** | `Health Insurance Fraud Claims.xlsx` (stored in `Approch-1/data/raw/` / root reference) | Shared single source of truth |
| **Cryptographic Digest (`SHA-256`)** | `952b6e23c9845b9086994c08bc81323575968c318f845ea69e3b9609614b6a45` | Ensures zero silent mutation between tracks |
| **Total Record Count ($N$)** | `4,500` claims | Divided into `3150/675/675` (random) or `3152/677/671` (temporal) |
| **Raw Column Width** | `19` columns (`18` predictors/keys/metadata + `1` target) | Covers financial, clinical, demographic, and cluster axes |
| **Target Distribution (`ClaimLegitimacy`)** | `270` Fraud (`1`) / `4,230` Legitimate (`0`) | Exact `6.00%` positive prevalence (`2` unique labels) |
| **Missingness Across All `19` Columns** | `0` nulls in every column (`4,500 * 19 = 85,500` complete cells) | No imputation transformations needed in pipeline |
| **Duplicate Row Count** | `0` duplicate records | Every claim row is unique |
| **Calendar Coverage (`ClaimDate`)** | `2022-07-09` to `2024-07-08` (`731` distinct dates) | Spans exactly two full leap-year calendar cycles (`2024` is leap year) |

### 3.2 Complete 19-Column Schema & Modeling Disposition Inventory

The table below catalogs every column in the `19`-column raw workbook along with its exact observed cardinality from `dataset_audit.json` and disposition across all modeling tracks:

| # | Raw Column Name | Unique Values (`dataset_audit.json`) | Missing Count | Semantic Role | Track A (`Approch-1`, `ml-service`) | Track B `intake` Policy | Track B `no_amount` Policy | Rationale for Modeling Disposition |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| `1` | `ClaimID` | `4,500` | `0` | Row identifier | Dropped | Dropped | Dropped | Pure per-claim UUID (`1:1` row cardinality); zero generalization value |
| `2` | `PatientID` | `4,500` | `0` | Patient identifier | Dropped | Dropped | Dropped | Unique per row (`4500/4500`); no repeat patient graph present |
| `3` | `ProviderID` | `4,500` | `0` | Provider identifier | Dropped | Dropped | Dropped | Unique per row (`4500/4500`); cannot build provider history |
| `4` | `ClaimAmount` | `4,490` | `0` | Financial claim value (`USD`) | **Included (`StandardScaler`)** | **Included (`StandardScaler`)** | **Dropped** | Primary monetary signal (`fraud min \$7,465.07`); dropped only in `no_amount` stress test |
| `5` | `ClaimDate` | `731` | `0` | Service / filing date | Transformed -> `Year, Month, DoW` | Transformed -> `Year, Month, DoW` | Transformed -> `Year, Month, DoW` | Raw timestamp parsed into `3` calendar integers; raw string dropped; splits `temporal` |
| `6` | `DiagnosisCode` | `4,495` | `0` | Clinical diagnosis string | Dropped | Dropped | Dropped | High cardinality (`99.89%` distinct across rows); sparse OHE explosion |
| `7` | `ProcedureCode` | `4,495` | `0` | Clinical procedure code | Dropped | Dropped | Dropped | High cardinality (`99.89%` distinct across rows); sparse OHE explosion |
| `8` | `PatientAge` | `100` | `0` | Claimant age (years) | **Included (`StandardScaler`)** | **Included (`StandardScaler`)** | **Included (`StandardScaler`)** | Continuous/discrete numeric predictor (`idx 1`) |
| `9` | `PatientGender` | `2` | `0` | Binary gender (`F`, `M`) | **Included (`OHE` -> `2` cols)** | **Included (`OHE` -> `2` cols)** | **Included (`OHE` -> `2` cols)** | Clean dichotomous demographic feature (`idx 7..8`) |
| `10` | `ProviderSpecialty` | `5` | `0` | Medical specialty | **Included (`OHE` -> `5` cols)** | **Included (`OHE` -> `5` cols)** | **Included (`OHE` -> `5` cols)** | `Cardiology`, `General Practice`, `Neurology`, `Orthopedics`, `Pediatrics` |
| `11` | `ClaimStatus` | `3` | `0` | Post-adjudication status | **Dropped (`LEAKY_COLUMNS`)** | **Dropped (`LEAKY_COLUMNS`)** | **Dropped (`LEAKY_COLUMNS`)** | Target leakage: reflects downstream adjudication decision unavailable pre-screening |
| `12` | `PatientIncome` | `4,500` | `0` | Reported annual income | **Included (`StandardScaler`)** | **Included (`StandardScaler`)** | **Included (`StandardScaler`)** | Continuous numeric demographic predictor (`idx 2`) |
| `13` | `PatientMaritalStatus` | `4` | `0` | Marital category | **Included (`OHE` -> `4` cols)** | **Included (`OHE` -> `4` cols)** | **Included (`OHE` -> `4` cols)** | `Divorced`, `Married`, `Single`, `Widowed` |
| `14` | `PatientEmploymentStatus` | `4` | `0` | Employment status | **Included (`OHE` -> `4` cols)** | **Included (`OHE` -> `4` cols)** | **Included (`OHE` -> `4` cols)** | `Employed`, `Retired`, `Student`, `Unemployed` |
| `15` | `ProviderLocation` | `3,876` | `0` | Geographic city/location | Dropped | Dropped | Dropped | `86.1%` unique categories; high noise & memorization hazard |
| `16` | `ClaimType` | `4` | `0` | Encounter modality | **Included (`OHE` -> `4` cols)** | **Included (`OHE` -> `4` cols)** | **Included (`OHE` -> `4` cols)** | `Emergency`, `Inpatient`, `Outpatient`, `Routine` |
| `17` | `ClaimSubmissionMethod` | `3` | `0` | Filing channel | **Included (`OHE` -> `3` cols)** | **Included (`OHE` -> `3` cols)** | **Included (`OHE` -> `3` cols)** | `Online`, `Paper`, `Phone` |
| `18` | `Cluster` | `4` | `0` | Unsupervised cluster ID (`0..3`) | **Included (`StandardScaler`)** | **Dropped (Post-Hoc Proxy)** | Policy-specific | `267/270` frauds reside in `Cluster 1`; dropped in `intake` (`ADR-07`) |
| `19` | `ClaimLegitimacy` | `2` | `0` | Binary supervised target | Target `y` (`Fraud=1, Legitimate=0`) | Target `y` (`Fraud=1, Legitimate=0`) | Target `y` (`Fraud=1, Legitimate=0`) | Ground-truth classification label (`270` positives / `4,230` negatives) |

Notice an extraordinary structural fact exposed by cardinality auditing: `ClaimID`, `PatientID`, `ProviderID`, and `PatientIncome` all have **`4,500` unique values across `4,500` rows**, while `DiagnosisCode` and `ProcedureCode` each have **`4,495` unique values**. Practically no repeated patient longitudinal history or recurring provider billing graph exists in this synthetic/curated benchmark corpus — justifying why row-level identifier dropping (`ClaimID`, `PatientID`, `ProviderID`, `DiagnosisCode`, `ProcedureCode`, `ProviderLocation`) is structurally necessary rather than merely convenient.

---

## 4. Quantitative Leakage, Proxy & Feature Dominance Proofs

Forensic auditing via `submission/results/dataset_audit.json` reveals two powerful latent data generation patterns that govern how every algorithm across Approaches 1, 2, and 3 behaves.

### 4.1 Proof of `Cluster` Post-Hoc Separation (`98.89%` Fraud Concentration)

Cross-tabulating target `ClaimLegitimacy` against the integer field `Cluster` across all `4,500` workbook claims yields:

| `Cluster` ID | Legitimate Count ($y=0$) | Fraud Count ($y=1$) | Total Cluster Size | Intra-Cluster Fraud Rate | Share of All `270` Frauds | Cumulative Fraud Share |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Cluster 0`** | `1,144` | `0` | `1,144` | `0.00%` | `0.00%` | `0.00%` |
| **`Cluster 1`** | `833` | **`267`** | `1,100` | **`24.27%`** | **`98.89%`** | `98.89%` |
| **`Cluster 2`** | `1,104` | `0` | `1,104` | `0.00%` | `0.00%` | `98.89%` |
| **`Cluster 3`** | `1,149` | `3` | `1,152` | `0.26%` | `1.11%` | `100.00%` |
| **Total Corpus** | **`4,230`** | **`270`** | **`4,500`** | **`6.00%`** | **`100.00%`** | — |

#### Operational Consequence for Real-Time Inference

1. **Single-Feature Gate Effect:** Any tree model allowed to split on `Cluster` immediately discards Clusters `0` and `2` (`2,248` legitimate claims, `50.0%` of total dataset) at `0` false-negative risk, isolating candidate search strictly inside `Cluster 1` (`24.27%` fraud prevalence — over `4x` corpus average!) plus just `3` edge cases in `Cluster 3`.
2. **Real-World Intake Infeasibility (`ADR-07`):** Because `Cluster` reflects a batch unsupervised segmentation tag derived over the curated workbook (not a raw claim intake field entered by a hospital billing clerk), expecting a live caller to supply an accurate `Cluster` integer prior to screening creates target-adjacent leakage risk.
3. **Why Track B Validates `intake` (28 Features):** Removing `Cluster` forces classifiers to reconstruct high-precision boundaries exclusively from causal claim amounts, demographic income/age correlations, calendar metadata, and categorical encounter attributes (`intake` policy).

### 4.2 Proof of `ClaimAmount` One-Sided Separation & Monetary Dominance

Descriptive monetary bounds in `submission/results/dataset_audit.json` establish a razor-sharp distributional asymmetry across `ClaimAmount`:

| Target Class | Row Count ($N$) | Minimum `ClaimAmount` | Maximum `ClaimAmount` | Arithmetic Mean `ClaimAmount` | Observed Regime Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Fraudulent (`y = 1`)** | `270` | **`\$7,465.07`** | `\$9,978.43` | `\$8,733.5651` | Strictly truncated inside upper `~25%` monetary band; zero frauds below `\$7,465.07` |
| **Legitimate (`y = 0`)** | `4,230` | `\$100.12` | **`\$9,997.20`** | `\$4,776.7978` | Uniform/broad coverage spanning `[\$100.12, \$9,997.20]`, overlapping fraud window cleanly |

Visualizing the `ClaimAmount` support along the horizontal dollar axis immediately explains why different model families exhibit starkly diverging behaviors:

```text
ClaimAmount ($ USD):
$100.12                                                         $7,465.07               $9,978.43  $9,997.20
|===================================================================|=======================|==========|
|-------------------- Legitimate Claims Only (0 Frauds) ------------|  OVERLAP ZONE:                   |
                                                                    |  - ALL 270 Frauds live here      |
                                                                    |  - Mean Fraud = $8,733.57        |
                                                                    |  - ~25% of Legit claims overlap! |
```

Three empirical phenomena observed across Approaches 1, 2, 3, and Track B drop straight out of this geometric structure:

1. **Why Logistic Regression Suffers `24` False Positives (`Precision = 62.50%`, `Recall = 100.00%`):** A linear half-space easily cuts near `ClaimAmount ≈ \$7,465` to achieve `100%` recall, yet lacks orthogonal conditional splits inside the overlap zone `[\$7,465.07, \$9,978.43]` to separate overlapping high-value legitimate inpatient/emergency bills (`Mean Legit = \$4,776.80`), sweeping legitimate high-amount claims into positive predictions.
2. **Why `Autoencoder` Fails Completely (`PR-AUC = 5.23%`):** Inside `[\$7,465.07, \$9,978.43]`, legitimate claims occur abundantly and share dense support with frauds under standardized $z$-scores (`ClaimAmount` standard deviation across `\$100–\$10,000` makes `\$8,733` sit merely ~`+1.3σ` above global mean `\$5,014`!). A low-rank linear/ReLU compressor (`29→32→10→32→29`) trained on all legitimate claims (`$100` to `$9,997`) reconstructs `\$8,733` claims with tiny error — often smaller MSE than legitimate outliers!
3. **Why Removing `ClaimAmount` (`no_amount`) Collapses Every Classifier (`AP = 0.1869–0.2404`):** Without `ClaimAmount`, neither decision trees nor neural nets can distinguish frauds from benign records beyond residual `Cluster`/categorical correlations, causing Average Precision to plunge from `~1.000` down to `0.2066` (`RandomForest`) and `0.1869` (`SVM`).
4. **Why Track B's Single Error (`Row 1124`) Occurs at Exact Boundary `\$7,465.07`:** As verified in Volume 07 (`analysis.json`), the **only** false negative committed by Track B's selected `intake` `RandomForest` (`FN=1` out of `40` test frauds) occurs at `source_row = 1124` whose `ClaimAmount` equals **`\$7,465.07`** — literally the minimum fraud monetary boundary in the entire dataset (`score = 0.27 < 0.4167`).

### 4.3 `ClaimStatus` Target Leakage (`LEAKY_COLUMNS`)

`ClaimStatus` carries `3` categorical adjudication states populated downstream in operational workflows. Both [`Approch-1/scripts/preprocess.py`](../../Approch-1/scripts/preprocess.py) and Track B (`submission/scripts/common.py`) declare `LEAKY_COLUMNS = ['ClaimStatus']` and purge it before splitting or fitting scalers, preventing retrospective status strings from masquerading as predictive signal.

---

## 5. Data Partitioning Topology & Cross-Track Equivalence Proofs

Both Track A and Track B enforce strict train-only parameter fitting (`StandardScaler` mean/variance and `OneHotEncoder` vocabulary fitted solely on $\mathcal{D}_{\text{train}}$, never across val/test) under two complementary splitting topologies:

### 5.1 Random Stratified `70 / 15 / 15` Split (`random_state = 42`)

Implemented first via two sequential `train_test_split(..., stratify=..., random_state=42)` calls in `Approch-1/scripts/preprocess.py` (`70%` train, `15%` val, `15%` test):

| Partition | Total Claims | Fraud Count ($y=1$) | Legitimate Count ($y=0$) | Exact Prevalence | Consumer Scripts |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`train`** | `3,150` | `189` | `2,961` | `6.000%` | `Approch-1/`, `Approch-2-DL/` (`X_train.npy`), `submission/` (`baseline`, `intake`, `no_amount`) |
| **`val`** | `675` | `41` | `634` | `6.074%` | Hyperparameter selection (`Best_Model`), PyTorch early stopping, Track B F1 threshold sweep |
| **`test`** | `675` | `40` | `635` | `5.926%` | Final held-out Track A metrics (`model_results.json`, `*_metrics.json`) & Track B evaluation |
| **Sum** | **`4,500`** | **`270`** | **`4,230`** | **`6.000%`** | Exhaustive non-overlapping partition |

#### Array Parity Verification (`dataset_audit.json`)

To certify that Track B reproduces the exact preprocessing arithmetic of Track A without silent numerical divergence, `submission/results/dataset_audit.json` compares Track B baseline matrices against `Approch-2-DL/processed/X_{train,val,test}.npy` at tight floating-point tolerance `atol = 1e-5`:

```json
"legacy_arrays_match": {
  "train": true,
  "val": true,
  "test": true
}
```

### 5.2 Chronological `temporal` Holdout (`submission/`)

Real fraud schemes drift across billing periods. Track B therefore defines an out-of-time split sorted monotonically by `ClaimDate` using date cutoffs `["2023-11-21", "2024-03-10"]` (`dataset_audit.json`):

| Chronological Partition | Date Interval | Total Claims | Fraud Count ($y=1$) | Legitimate Count ($y=0$) | Temporal Prevalence | Track B Imbalance Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`temporal_train`** | `2022-07-09` to `2023-11-20` | `3,152` | `199` | `2,953` | `6.313%` | `2953 / 199 ≈ 14.8392` |
| **`temporal_val`** | `2023-11-21` to `2024-03-10` | `677` | `34` | `643` | `5.022%` | — |
| **`temporal_test`** | `2024-03-11` to `2024-07-08` | `671` | `37` | `634` | `5.514%` | — |
| **Sum** | `2022-07-09` to `2024-07-08` | **`4,500`** | **`270`** | **`4,230`** | **`6.000%`** | Zero future leakage |

---

## 6. Complete Canonical 29-Feature Contract (`feature_names.txt`)

Every Track A serialized artifact (`Approch-1/models/*.joblib`, `Approch-2-DL/models/*_best.pt`, `ml-service/app/services/*.py`) depends strictly on the positional column ordering locked in [`Approch-1/data/processed/feature_names.txt`](../../Approch-1/data/processed/feature_names.txt) (lines `1–29`) and summarized in [`Approch-2-DL/processed/contract_metadata.json`](../../Approch-2-DL/processed/contract_metadata.json).

### 6.1 Exact 29-Feature Index Mapping Table

| Vector Index | Canonical Feature Name (`feature_names.txt`) | Transformer Component | Source Workbook Field | Intake (`28f`) Status | Live FastAPI Input Field (`claims.py`) |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **`0`** | `ClaimAmount` | `StandardScaler` (`num[0]`) | `ClaimAmount` | Kept (`idx 0`) | `ClaimAmount: float` |
| **`1`** | `PatientAge` | `StandardScaler` (`num[1]`) | `PatientAge` | Kept (`idx 1`) | `PatientAge: int` |
| **`2`** | `PatientIncome` | `StandardScaler` (`num[2]`) | `PatientIncome` | Kept (`idx 2`) | `PatientIncome: float` |
| **`3`** | `ClaimYear` | `StandardScaler` (`num[3]`) | `ClaimDate.dt.year` | Kept (`idx 3`) | Derived from `ClaimDate` (`YYYY-MM-DD`) |
| **`4`** | `ClaimMonth` | `StandardScaler` (`num[4]`) | `ClaimDate.dt.month` | Kept (`idx 4`) | Derived from `ClaimDate` (`1..12`) |
| **`5`** | `ClaimDayOfWeek` | `StandardScaler` (`num[5]`) | `ClaimDate.dt.dayofweek` | Kept (`idx 5`) | Derived from `ClaimDate` (`Mon=0..Sun=6`) |
| **`6`** | `Cluster` | `StandardScaler` (`num[6]`) | `Cluster` | **Excluded** | `Cluster: int` |
| **`7`** | `PatientGender_F` | `OneHotEncoder` (`cat[0]`) | `PatientGender == "F"` | Kept (`idx 6`) | `PatientGender: str` |
| **`8`** | `PatientGender_M` | `OneHotEncoder` (`cat[1]`) | `PatientGender == "M"` | Kept (`idx 7`) | `PatientGender: str` |
| **`9`** | `ProviderSpecialty_Cardiology` | `OneHotEncoder` (`cat[2]`) | `ProviderSpecialty == "Cardiology"` | Kept (`idx 8`) | `ProviderSpecialty: str` |
| **`10`** | `ProviderSpecialty_General Practice` | `OneHotEncoder` (`cat[3]`) | `ProviderSpecialty == "General Practice"` | Kept (`idx 9`) | `ProviderSpecialty: str` |
| **`11`** | `ProviderSpecialty_Neurology` | `OneHotEncoder` (`cat[4]`) | `ProviderSpecialty == "Neurology"` | Kept (`idx 10`) | `ProviderSpecialty: str` |
| **`12`** | `ProviderSpecialty_Orthopedics` | `OneHotEncoder` (`cat[5]`) | `ProviderSpecialty == "Orthopedics"` | Kept (`idx 11`) | `ProviderSpecialty: str` |
| **`13`** | `ProviderSpecialty_Pediatrics` | `OneHotEncoder` (`cat[6]`) | `ProviderSpecialty == "Pediatrics"` | Kept (`idx 12`) | `ProviderSpecialty: str` |
| **`14`** | `PatientMaritalStatus_Divorced` | `OneHotEncoder` (`cat[7]`) | `PatientMaritalStatus == "Divorced"` | Kept (`idx 13`) | `PatientMaritalStatus: str` |
| **`15`** | `PatientMaritalStatus_Married` | `OneHotEncoder` (`cat[8]`) | `PatientMaritalStatus == "Married"` | Kept (`idx 14`) | `PatientMaritalStatus: str` |
| **`16`** | `PatientMaritalStatus_Single` | `OneHotEncoder` (`cat[9]`) | `PatientMaritalStatus == "Single"` | Kept (`idx 15`) | `PatientMaritalStatus: str` |
| **`17`** | `PatientMaritalStatus_Widowed` | `OneHotEncoder` (`cat[10]`) | `PatientMaritalStatus == "Widowed"` | Kept (`idx 16`) | `PatientMaritalStatus: str` |
| **`18`** | `PatientEmploymentStatus_Employed` | `OneHotEncoder` (`cat[11]`) | `PatientEmploymentStatus == "Employed"` | Kept (`idx 17`) | `PatientEmploymentStatus: str` |
| **`19`** | `PatientEmploymentStatus_Retired` | `OneHotEncoder` (`cat[12]`) | `PatientEmploymentStatus == "Retired"` | Kept (`idx 18`) | `PatientEmploymentStatus: str` |
| **`20`** | `PatientEmploymentStatus_Student` | `OneHotEncoder` (`cat[13]`) | `PatientEmploymentStatus == "Student"` | Kept (`idx 19`) | `PatientEmploymentStatus: str` |
| **`21`** | `PatientEmploymentStatus_Unemployed` | `OneHotEncoder` (`cat[14]`) | `PatientEmploymentStatus == "Unemployed"` | Kept (`idx 20`) | `PatientEmploymentStatus: str` |
| **`22`** | `ClaimType_Emergency` | `OneHotEncoder` (`cat[15]`) | `ClaimType == "Emergency"` | Kept (`idx 21`) | `ClaimType: str` |
| **`23`** | `ClaimType_Inpatient` | `OneHotEncoder` (`cat[16]`) | `ClaimType == "Inpatient"` | Kept (`idx 22`) | `ClaimType: str` |
| **`24`** | `ClaimType_Outpatient` | `OneHotEncoder` (`cat[17]`) | `ClaimType == "Outpatient"` | Kept (`idx 23`) | `ClaimType: str` |
| **`25`** | `ClaimType_Routine` | `OneHotEncoder` (`cat[18]`) | `ClaimType == "Routine"` | Kept (`idx 24`) | `ClaimType: str` |
| **`26`** | `ClaimSubmissionMethod_Online` | `OneHotEncoder` (`cat[19]`) | `ClaimSubmissionMethod == "Online"` | Kept (`idx 25`) | `ClaimSubmissionMethod: str` |
| **`27`** | `ClaimSubmissionMethod_Paper` | `OneHotEncoder` (`cat[20]`) | `ClaimSubmissionMethod == "Paper"` | Kept (`idx 26`) | `ClaimSubmissionMethod: str` |
| **`28`** | `ClaimSubmissionMethod_Phone` | `OneHotEncoder` (`cat[21]`) | `ClaimSubmissionMethod == "Phone"` | Kept (`idx 27`) | `ClaimSubmissionMethod: str` |

### 6.2 ColumnTransformer Configuration & Unknown-Category Guard

Both `Approch-1/scripts/preprocess.py` and `ml-service` operate over an identically configured scikit-learn `ColumnTransformer`:

* **Numeric Pipeline:** `StandardScaler()` applied strictly to indices `[0..6]` (`z = (x - mu_train) / sigma_train`). Notice `float64` NumPy outputs from `preprocess.py` are down-cast safely to `float32` (`torch.float32`) when entering Approach 2 PyTorch `DataLoader` tensors.
* **Categorical Pipeline:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to the `6` categorical columns (`indices 7..28`). Using `handle_unknown='ignore'` guarantees runtime safety: if an unseen category arrives in a test batch or live JSON request, the corresponding block outputs an all-zero slice instead of raising a `ValueError` exception (meanwhile `ClaimRequest` with `extra="forbid"` rejects undeclared JSON *property keys* cleanly with HTTP `422`).

### 6.3 Worked End-to-End Claim Vectorization Walkthrough

To demonstrate exact mechanical determinacy across Approach 1 `.joblib`, Approach 2 `.pt`, and FastAPI `/predict`, trace how a concrete claim passes from JSON intake to the `(1, 29)` inference tensor:

#### Step A: Inbound Raw JSON Request (`ml-service` `ClaimRequest`)

```json
{
  "ClaimAmount": 7465.07,
  "PatientAge": 39,
  "PatientIncome": 62500.0,
  "ClaimDate": "2024-03-19",
  "Cluster": 1,
  "PatientGender": "F",
  "ProviderSpecialty": "Cardiology",
  "PatientMaritalStatus": "Married",
  "PatientEmploymentStatus": "Employed",
  "ClaimType": "Outpatient",
  "ClaimSubmissionMethod": "Online"
}
```

#### Step B: Date Parsing & Numeric Alignment (`indices 0..6`)

`pd.to_datetime("2024-03-19")` extracts calendar integers following Python/pandas ISO convention (`Monday = 0` .. `Sunday = 6`; `2024-03-19` is Tuesday -> `ClaimDayOfWeek = 1`):

* `num_raw = [7465.07, 39, 62500.0, 2024, 3, 1, 1]`
* `StandardScaler.transform` applies $(x_j - \mu_j) / \sigma_j$ using saved `preprocessor.joblib` statistics fitted exclusively on `N = 3,150` training rows, placing `ClaimAmount = 7465.07` strongly positive relative to legitimate mean `\$4,776.80`.

#### Step C: One-Hot Activation Vector (`indices 7..28`, `22` Positions)

Exactly one bit flips to `1.0` inside each of the six alphabetical categorical groups (`[0] * 22` everywhere else):

* `PatientGender = "F"` → `idx 7 = 1.0`, `idx 8 = 0.0` → `[1, 0]`
* `ProviderSpecialty = "Cardiology"` → `idx 9 = 1.0`, `idx 10..13 = 0.0` → `[1, 0, 0, 0, 0]`
* `PatientMaritalStatus = "Married"` → `idx 15 = 1.0`, `idx {14,16,17} = 0.0` → `[0, 1, 0, 0]`
* `PatientEmploymentStatus = "Employed"` → `idx 18 = 1.0`, `idx 19..21 = 0.0` → `[1, 0, 0, 0]`
* `ClaimType = "Outpatient"` → `idx 24 = 1.0`, `idx {22,23,25} = 0.0` → `[0, 0, 1, 0]`
* `ClaimSubmissionMethod = "Online"` → `idx 26 = 1.0`, `idx {27,28} = 0.0` → `[1, 0, 0]`

#### Step D: Final Routing Across Model Families

1. **Classical ML (`classical_service.py`):** Receives `(1, 29)` array (preserving feature frame column names or ordered float array). `XGBClassifier`, `RandomForestClassifier`, `HistGradientBoostingClassifier`, and `LogisticRegression` compute `.predict_proba(X)[:, 1]`. Note carefully that `CNN1D` and `TabTransformer` rely explicitly on this **fixed index adjacency**: changing column positions in `feature_names.txt` would scramble local `kernel_size=3` receptive fields in `CNN1D` (`idx 0..2`, `1..3`, etc.) unless retrained cleanly.
2. **Deep Learning (`dl_service.py`):** Converts vector to `torch.float32` tensor shaped `(1, 29)`:
   * `MLP` / `ResidualMLP` / `Autoencoder` consume `(1, 29)` directly.
   * `CNN1D` permutes view inside `forward()` via `x.unsqueeze(1)` → `(1, 1, 29)`.
   * `TabTransformer` reshapes inside `forward()` via `x.unsqueeze(-1)` → `(1, 29, 1)` before `Linear(1, 32)` → `(1, 29, 32)`.
3. **Browser Pure-JS Track B (`export_browser.py`):** Replicates Steps B–D deterministically in vanilla JavaScript (`Float64`/`Float32` dot products + serialized mean/scale vectors + categorical lookup tables) achieving `max_err = 0.0` on all `1,350` val+test claims (`browser_parity.json`).

---

## 7. Cross-Track Contract Invariants & Validation Checklist

Every data engineer modifying preprocessing or inference schemas across this workspace must preserve five non-negotiable invariants verified across `dataset_audit.json`, `contract_metadata.json`, and `ml-service`:

| Invariant ID | Formal Rule | What Breaks if Violated | Automated Guard in Repository |
| :--- | :--- | :--- | :--- |
| **INV-01** | **Train-Only Fit:** `StandardScaler` means/stds and `OneHotEncoder` categories MUST be fitted strictly on `train` splits. | Data leakage between test distribution and scaling statistics; violates `legacy_arrays_match`. | `preprocess.py` & `submission/scripts/common.py` |
| **INV-02** | **Exact Alphabetical OHE Ordering:** Categorical indicators preserve scikit-learn's lexicographical category sort (`F, M`; `Cardiology..Pediatrics`; `Divorced..Widowed`; `Employed..Unemployed`; `Emergency..Routine`; `Online..Phone`). | Silently permutes OHE bits for saved `MLP`, `CNN1D`, `TabTransformer`, and tree checkpoints, inverting predictions! | `feature_names.txt` (`1..29`) & `contract.json` |
| **INV-03** | **`ClaimStatus` Never Enters Predictors:** Dropped unconditionally prior to pipeline branching. | Inflates scores via post-adjudication target leakage (`LEAKY_COLUMNS`). | Explicit column drop in all preprocessors |
| **INV-04** | **`ClaimDayOfWeek` Zero-Based Monday Indexing:** Must follow pandas `.dt.dayofweek` (`Monday=0 .. Sunday=6`), NOT SQL `DAYOFWEEK` (`1..7`) or JS `getDay()` (`Sunday=0`) without mapping! | Shifts `ClaimDayOfWeek` `StandardScaler` inputs by $\pm 1$ day when exporting Python logic to Node/JS. | `export_browser.py` parity suite (`max_err = 0` over `1,350` rows) |
| **INV-05** | **Strict Pydantic Body Rejections:** Unknown JSON attributes must trigger HTTP `422`. | Silent typo acceptance producing zero-padded fraudulent/legit predictions. | `ConfigDict(extra="forbid")` in `claims.py`; `4` negative tests in `verification.json` |

---

## 8. Summary Transition to Implementation Volumes

* With mathematical optimization objectives, PR-AUC theory, 19-column SHA-256 workbook forensics, `Cluster`/`ClaimAmount` separation proofs, and the canonical 29-feature schema locked down, we inspect each concrete AI approach sequentially:
  * [**Volume 03 — Approach 1: Classical Machine Learning (`Approch-1/`)**](./03_APPROACH_1_CLASSICAL_MACHINE_LEARNING.md)
  * [**Volume 04 — Approach 2: Deep Learning (`Approch-2-DL/`)**](./04_APPROACH_2_DEEP_LEARNING.md)
  * [**Volume 05 — Approach 3: Agentic AI, XAI & Copilot**](./05_APPROACH_3_AGENTIC_AI_XAI_AND_COPILOT.md)
