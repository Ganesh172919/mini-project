# Problem Formulation: Medical Insurance Claim Fraud Detection

**Project:** Health Insurance Fraud Intelligence Platform (`Medical Insurance Claim Fraud Detection`)  
**Dataset:** `Health Insurance Fraud Claims.xlsx` (SHA-256: `952b6e23c9845b9086994c08bc81323575968c318f845ea69e3b9609614b6a45`)  
**Date:** November 2026

---

## 1. Domain Problem Statement

Healthcare insurance fraud entails deceptive billing behaviors—including upcoding, phantom billing, medically unnecessary procedures, and identity misuse—submitted for reimbursement under outpatient, inpatient, routine, or emergency care claims. Manual auditing by a Special Investigation Unit (SIU) is expensive and slow; unconstrained automated blocking risks wrongful patient claim denial. The engineering objective is therefore to construct an **auditable, multi-paradigm claim triage system** that ranks suspicious claims accurately under severe class imbalance ($6.00\%$ positive rate), exposes leakage and single-feature shortcuts systematically, and explains flagged predictions to human investigators.

---

## 2. Dataset Schema and Audited Distributional Structure

Let $\mathcal{D} = \{(\mathbf{r}_i, y_i)\}_{i=1}^{N}$ denote the raw corpus of $N = 4,500$ claims with $19$ attributes observed on disk (`0` missing values, `0` duplicate rows, spanning 731 calendar days from `2022-07-09` to `2024-07-08`), where the binary fraud indicator is derived as:

$$
y_i = \mathbb{I}(\text{ClaimLegitimacy}_i = \text{``Fraud''}) \in \{0, 1\}, \quad N_+ = \sum_{i=1}^N y_i = 270 \ (6.00\%), \quad N_- = 4,230 \ (94.00\%).
$$

The 19 raw columns partition naturally into four functional groups:

1. **Identifiers & High-Cardinality Codes (Excluded from Baseline Model Matrix):** `ClaimID` (4,500 unique), `PatientID` (4,500), `ProviderID` (4,500), `DiagnosisCode` (4,495), `ProcedureCode` (4,495), `ProviderLocation` (3,876). Each acts nearly as a unique record key; naive inclusion causes severe overfitting or out-of-vocabulary failure.
2. **Numerical & Temporal Covariates:** `ClaimAmount` ($\min=100.12$, $\text{median}=5,053.77$, $\max=9,997.20$), `PatientAge` ($0\text{--}99$, $\text{median}=50.5$), `PatientIncome` ($20,006.87\text{--}149,957.52$), and `ClaimDate` decomposed into `ClaimYear`, `ClaimMonth`, `ClaimDayOfWeek`.
3. **Clinical & Administrative Categoricals:** `PatientGender` (2), `ProviderSpecialty` (5), `ClaimStatus` (3: Approved, Pending, Denied), `PatientMaritalStatus` (4), `PatientEmploymentStatus` (4), `ClaimType` (4: Outpatient, Inpatient, Routine, Emergency), `ClaimSubmissionMethod` (3: Online, Paper, Phone).
4. **Upstream Segmentation Feature (`Cluster`):** Integer $\{0,1,2,3\}$. Audited cross-tabulation reveals **267 of 270 frauds (98.89%) lie inside `Cluster = 1`** ($267/1,100 = 24.27\%$ fraud rate vs. $0/1,144$ in Cluster 0, $0/1,104$ in Cluster 2, and $3/1,152 = 0.26\%$ in Cluster 3). Treating `Cluster` as an unexamined runtime predictor creates **upstream proxy leakage**.
5. **Distributional Range Finding on `ClaimAmount`:** Every fraudulent claim satisfies $\text{ClaimAmount}_i \ge 7,465.07$ ($\text{mean} = 8,733.57$, $\max = 9,978.43$), whereas legitimate claims span the full domain $[100.12, 9,997.20]$ ($\text{mean} = 4,776.80$). Evaluating robustness *without* `ClaimAmount` is therefore critical to detect single-feature shortcut dependence.

---

## 3. Formal Mathematical Tasks & Experimental Feature Contracts

To resolve tension between full-stack backward compatibility (29 features) and deployment realism (no upstream `Cluster`), we formalize **four controlled feature/split contracts** $\phi^{(s)}(\mathbf{r}_i) \in \mathbb{R}^{d_s}$ fitted strictly on training split $\mathcal{D}_{\text{train}}$ (standard scaling via $\mu_j, \sigma_j$ computed on $\mathcal{D}_{\text{train}}$ only, plus one-hot encoding with `handle_unknown='ignore'`):

| Contract $s$ | Dim $d_s$ | Split Policy | Split Sizes ($N_{\text{train}}/N_{\text{val}}/N_{\text{test}}$, Fraud Counts) | Purpose & Track Mapping |
|---|---|---|---|---|
| **Baseline** (`baseline`) | $29$ | Stratified Random 70/15/15 (Seed `42`) | $3,150$ (`189`) / $675$ (`41`) / $675$ (`40`) | Legacy full-stack Track A (10 models) & Track B baseline anchor (includes `Cluster`). |
| **Intake** (`intake`) | $28$ | Stratified Random 70/15/15 (Seed `42`) | $3,150$ (`189`) / $675$ (`41`) / $675$ (`40`) | **Deployment-realistic contract**: excludes `Cluster` (`28 = 6 numeric + 22 one-hot`). |
| **No-Amount** (`no_amount`) | $27$ | Stratified Random 70/15/15 (Seed `42`) | $3,150$ (`189`) / $675$ (`41`) / $675$ (`40`) | **Shortcut stress test**: drops both `Cluster` and `ClaimAmount` (`5 numeric + 22 one-hot`). |
| **Temporal Intake** (`temporal`) | $28$ | Chronological Date Split (`<=2023-11-21` / `<=2024-03-10` / `>=2024-03-11`) | $3,152$ (`199`) / $677$ (`34`) / $671$ (`37`) | **Forward-in-time generalization**: eliminates random temporal mixing across time. |

### 3.1 Task 1: Imbalance-Aware Supervised Binary Risk Scoring (Approaches 1 & 2)
Given transformed vector $\mathbf{x}_i = \phi^{(s)}(\mathbf{r}_i) \in \mathbb{R}^{d_s}$, learn a parametric or ensemble scoring function $f_\theta: \mathbb{R}^{d_s} \to [0, 1]$ estimating uncalibrated relative fraud propensity $p_i = f_\theta(\mathbf{x}_i)$ by minimizing imbalance-weighted empirical risk on $\mathcal{D}_{\text{train}}$:

- **Weighted Binary Cross-Entropy (Supervised Deep Learning: MLP, ResidualMLP, CNN1D, TabTransformer):**

$$
\mathcal{L}_{\text{wBCE}}(\theta) = -\frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \mathcal{D}_{\text{train}}} \left[ w^+ y_i \log \sigma(z_i) + (1 - y_i)\log(1 - \sigma(z_i)) \right], \quad w^+ = \frac{N_{-,\text{train}}}{N_{+,\text{train}}} \approx 15.67,
$$

where $z_i = g_\theta(\mathbf{x}_i)$ is the raw logit and $\sigma(\cdot)$ is the logistic sigmoid, paired with AdamW weight decay and early stopping monitored via validation **PR-AUC**.
- **Classical Ensembles & Linear/Kernel Baselines:** `LogisticRegression(class_weight="balanced")`, `RandomForestClassifier(200, balanced)`, `SVC(RBF, class_weight="balanced")`, `XGBClassifier(scale_pos_weight=w^+)`, and `HistGradientBoostingClassifier(max_iter=200, early_stopping=False)`.

### 3.2 Task 2: Unsupervised Reconstruction Anomaly Detection (Autoencoder Ablation)
Train encoder-decoder network $(E_\psi, D_\psi)$ ($29 \to 16 \to 8 \to 16 \to 29$) **exclusively on legitimate training claims** $\mathcal{D}_{\text{train}}^{-} = \{\mathbf{x}_i \in \mathcal{D}_{\text{train}} : y_i = 0\}$ to minimize mean-squared reconstruction error:

$$
\mathcal{L}_{\text{AE}}(\psi) = \frac{1}{|\mathcal{D}_{\text{train}}^{-}|} \sum_{i \in \mathcal{D}_{\text{train}}^{-}} \|\mathbf{x}_i - D_\psi(E_\psi(\mathbf{x}_i))\|_2^2, \quad s_{\text{AE}}(\mathbf{x}) = \|\mathbf{x} - D_\psi(E_\psi(\mathbf{x}))\|_2^2.
$$

Decision cutoff $\tau_{\text{AE}}$ is selected unsupervisedly at the 95th percentile of validation legitimate/overall reconstruction errors ($\tau_{\text{AE}} = 0.128588$). Because fraud cases lie mostly inside the dense legitimate feature bounding box ($[7,465.07, 9,978.43] \subset [100.12, 9,997.20]$), high reconstruction fidelity on frauds becomes an empirical test of manifold overlap vs. separability.

### 3.3 Task 3: Operating Threshold Optimization & Evaluation Protocol
With only $6\%$ positive prevalence, accuracy ($94\%$ baseline for an all-negative dummy classifier) and `ROC-AUC` mask false-positive alert load. Therefore:
1. **Primary Ranking Metric:** **Precision-Recall Area Under Curve (`PR-AUC` / Average Precision `AP`)**.
2. **Operating Decision Function:** $\hat{y}_i(\tau) = \mathbb{I}(f_\theta(\mathbf{x}_i) \ge \tau)$. Whereas Track A reports standard $\tau = 0.50$ metrics alongside `ROC-AUC` and `PR-AUC`, Track B tunes the decision boundary strictly on the **validation set** to avoid test leakage:

$$
\tau^* = \arg\max_{\tau \in \{f_\theta(\mathbf{x}) : \mathbf{x} \in \mathcal{D}_{\text{val}}\}} F_1\big(\mathbf{y}_{\text{val}}, \mathbb{I}(f_\theta(\mathbf{X}_{\text{val}}) \ge \tau)\big), \quad \text{evaluated frozen on } \mathcal{D}_{\text{test}}.
$$

3. **Uncertainty Quantification:** Non-parametric stratified bootstrap ($B = 1,000$ test resamples, fixed model & $\tau^*$, seed `2026`) yields 95% confidence intervals for `AP`, `F1`, and `Recall`.

### 3.4 Task 4: Agentic Triage, Explainability, and Dual-Path Parity Contract
- **Agentic Audit Trail & XAI (Approach 3):** Transform structured claim attributes and model outputs into ordered stage logs $(\text{Stage}_k, \text{Status}_k, \text{Finding}_k)$ accompanied by deterministic per-feature directional attributions, cryptographic claim-log integrity digests (`SHA-256`), and interactive Gemini co-pilot reasoning (`/api/chat`).
- **Dual-Path Parity Constraint:** For any browser-deployed exported weight bundle in Track B (`submission/`), enforce zero-mismatch alignment between Python training/inference code and JavaScript browser inference:

$$
\max_{i \in \mathcal{D}_{\text{test}} \cup \mathcal{D}_{\text{val}},\, m} \left| f^{\text{Python}}_m(\mathbf{x}_i) - f^{\text{Browser}}_m(\mathbf{x}_i) \right| < 10^{-6}, \quad \sum \left| \hat{y}^{\text{Python}}_{m,i} - \hat{y}^{\text{Browser}}_{m,i} \right| = 0.
$$
