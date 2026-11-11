# Volume 10 — Model Cards: Approach 2 Deep Learning Portfolio (`Approch-2-DL/` & `submission/`)

> **Document Classification:** Standardized Neural Architecture Model Cards (`5` PyTorch Topologies)  
> **Primary Ownership:** **M. Jagadeeswar (`23BDS033`)** (Track A neural suite & FastAPI serving) + **J. Ganesh Kumar Reddy (`23BDS024`)** (Track B deterministic re-evaluation & pure-JS browser export)  
> **Primary Ground-Truth Sources:** [`Approch-2-DL/scripts/dl_common.py`](../../Approch-2-DL/scripts/dl_common.py), [`submission_models.py`](../../Approch-2-DL/scripts/submission_models.py), [`dl_advanced_models.py`](../../Approch-2-DL/scripts/dl_advanced_models.py), [`Approch-2-DL/results/*_metrics.json`](../../Approch-2-DL/results/mlp_metrics.json), [`logs/training_history.json`](../../Approch-2-DL/logs/training_history.json), [`ml-service/app/services/dl_service.py`](../../ml-service/app/services/dl_service.py), and [`submission/results/baseline_ResidualMLP_training.json`](../../submission/results/baseline_ResidualMLP_training.json).

---

## 1. Portfolio Summary Across All Five Neural Architectures

All supervised Deep Learning models train under identical optimization rules (`AdamW(lr=1e-3, weight_decay=1e-4)`, `batch_size=128`, `BCEWithLogitsLoss(pos_weight=2961/189)`, early stopping `patience=15` monitoring `val_pr_auc`, `seed=42`) over Track A's canonical `(3150 / 675 / 675) x 29` tensors. `Autoencoder` runs unsupervised benign reconstruction over `y == 0` (`N = 2,961`).

| Model Card ID | Disk Checkpoint (`Approch-2-DL/models/`) | Inductive Class & Module Source | Param Count | Stop Epoch | Track A Test PR-AUC | Track A Test F1 | Checkpoint Bytes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`CARD-DL-01`** | `mlp_best.pt` | `MLP` (`dl_common.py` L50–64, **NO BatchNorm**) | `4,033` | `34 / 49` | `95.38%` | `90.70%` | `19,873 B` |
| **`CARD-DL-02`** | `residualmlp_best.pt` | `ResidualMLP` (`submission_models.py` L5–19) | `18,625` | `20` *(B: `42`)* | `95.61%` *(B: `95.67%`)* | `88.89%` | `80,161 B` |
| **`CARD-DL-03`** | `cnn1d_best.pt` | `CNN1D` (`submission_models.py` L21–27) | `15,345` | `41 / 56` | `98.82%` | **`90.91%`** | `65,591 B` |
| **`CARD-DL-04`** | `tabtransformer_best.pt` | `TabTransformer` (`dl_advanced_models.py` L16–57) | Modular 2L Transf. | `34` | **`99.11%`** | `89.89%` | `351,993 B` |
| **`CARD-DL-05`** | `autoencoder_best.pt` | `Autoencoder` (`dl_advanced_models.py` L60–116) | `2,599` | `35` *(unsup)* | `5.23%` *(neg. result)* | `2.99%` | `14,411 B` |

---

## 2. Model Card `CARD-DL-01`: Baseline Feed-Forward `MLP`

### 2.1 Card Identity, Topology & Exact Parameter Arithmetic

* **Source Definition:** `MLP` in [`Approch-2-DL/scripts/dl_common.py`](../../Approch-2-DL/scripts/dl_common.py) L50–64.
* **Checkpoint / Artifact Size:** `mlp_best.pt` (`19,873` bytes) — compact lightweight neural classifier (~`4×` smaller than `ResidualMLP`, ~`18×` smaller than `TabTransformer`).
* **Architectural Verification Note:** Contains **no `BatchNorm1d` layers** (rebutting inaccurate claims in prior `docs/03`).

| Layer Stage | Verified Tensor Operation | Input Shape | Output Shape | Exact Weights + Biases |
| :---: | :--- | :---: | :---: | :---: |
| `L1` | `Linear(29, 64)` → `ReLU` → `Dropout(0.2)` | `(B, 29)` | `(B, 64)` | `29 × 64 + 64 = 1,920` |
| `L2` | `Linear(64, 32)` → `ReLU` → `Dropout(0.2)` | `(B, 64)` | `(B, 32)` | `64 × 32 + 32 = 2,080` |
| `L3` | `Linear(32, 1)` *(raw logit $z$)* | `(B, 32)` | `(B, 1)` | `32 × 1 + 1 = 33` |
| **Sum** | **Full Three-Layer Feed-Forward Net** | `(B, 29)` | `(B, 1)` | **`4,033` Trainable Parameters** |

### 2.2 Optimization Trajectory & Empirical Profile

* **Training History (`train_mlp.py` → `logs/training_history.json`):** Executed `49` epochs total in **`4.43s`**, selecting peak `val_pr_auc` checkpoint at **Epoch `34`** (`patience = 15`).
* **Track A Held-Out Test (`N = 675`, `40` Frauds):**
  * Accuracy: **`98.81%`** | Precision: **`84.78%`** *(Best DL Precision!)* | Recall: `97.50%` | F1: `90.70%`
  * ROC-AUC: `99.73%` | PR-AUC: `95.38%` | Batch Latency: `0.732 ms`
  * Confusion Counts: **`TN = 628`**, **`FP = 7`**, **`FN = 1`**, **`TP = 39`**
* **Derived Operating Diagnostics (`(628, 7, 1, 39)`):**
  * Specificity (`TNR`): **`98.90%`** | `FPR`: **`1.10%`** | `FNR`: `2.50%` | `FDR`: `15.22%` | `NPV`: `99.84%` | `MCC`: `0.9032`.
* **Track B `temporal` Holdout (`>= 2024-03-11`):** Drops to `AP = 0.8581`, `F1 = 0.6957` (`14.19` AP points behind deterministic trees due to unregularized continuous weight response to `ClaimYear = 2024` shift).

### 2.3 Failure Mode & Calibration Reality

At only `4,033` parameters (`~1.28` parameters per training sample!), `MLP` resists extreme variance better than wide networks on precision (`FP = 7`, lowest in DL), yet its logit bias shifts by $+\ln(15.6667) \approx +2.7515$ (`score_is_calibrated = False`), triggering `7` benign false positives at default `0.50` probability cutoff.

---

## 3. Model Card `CARD-DL-02`: Skip-Connected `ResidualMLP`

### 3.1 Card Identity, Topology & Exact Parameter Arithmetic

* **Source Definition:** `ResidualBlock` + `ResidualMLP` in [`Approch-2-DL/scripts/submission_models.py`](../../Approch-2-DL/scripts/submission_models.py) L5–19.
* **Checkpoint / Artifact Size:** `residualmlp_best.pt` (`80,161` bytes).

| Block / Layer | Verified Operations | Input Shape | Output Shape | Exact Parameter Count |
| :---: | :--- | :---: | :---: | :---: |
| **Input Stem** | `Linear(29, 64)` → `ReLU` | `(B, 29)` | `(B, 64)` | `29 × 64 + 64 = 1,920` |
| **ResBlock `1`** | `x + [Linear(64,64) → ReLU → Dropout(0.2) → Linear(64,64)]` then `.relu()` | `(B, 64)` | `(B, 64)` | `2 × (64 × 64 + 64) = 8,320` |
| **ResBlock `2`** | `x + [Linear(64,64) → ReLU → Dropout(0.2) → Linear(64,64)]` then `.relu()` | `(B, 64)` | `(B, 64)` | `2 × (64 × 64 + 64) = 8,320` |
| **Logit Head** | `Linear(64, 1)` | `(B, 64)` | `(B, 1)` | `64 × 1 + 1 = 65` |
| **Sum** | **Two-Block Tabular Residual Highway** | `(B, 29)` | `(B, 1)` | **`18,625` Trainable Parameters** |

### 3.2 Track A vs. Track B Trajectory Divergence

1. **Track A Execution (`train_supervised.py`):** Converged rapidly at **Epoch `20`** (`0.674 ms` fastest supervised neural batch inference!):
   * Test Accuracy: `98.52%` | Precision: `80.00%` | Recall: **`100.00%`** | F1: `88.89%` | ROC: `99.75%` | PR-AUC: `95.61%`
   * Confusion Counts: **`TN = 625`**, **`FP = 10`**, **`FN = 0`**, **`TP = 40`**
   * Derived Rates: Specificity `98.43%`, `FPR = 1.57%`, `FNR = 0.00%`, `FDR = 20.00%`, `NPV = 100.00%`, `MCC = 0.8874`.
2. **Track B Deterministic Execution (`torch.use_deterministic_algorithms(True)`, `baseline_ResidualMLP_training.json`):** Deterministic gradient reduction altered optimization path cleanly, converging at **Epoch `42`** (`4.42s`) with **`Test PR-AUC = 95.67%`**.

### 3.3 Severe Temporal Extrapolation Vulnerability

Under Track B's out-of-time `temporal` benchmark (`N_test = 671`), `ResidualMLP` suffered the **steepest chronological decay of any evaluated classifier (`AP = 0.7979`, `F1 = 0.7229`)**. Because residual skip paths pass scaled `Linear(29, 64)` representations directly into downstream additions without strong bounded gating, out-of-distribution `ClaimYear` coordinates (`2024`) propagate linearly through both highway blocks, degrading precision on later months.

---

## 4. Model Card `CARD-DL-03`: Spatial Feature-Window `CNN1D`

### 4.1 Card Identity, Topology & Exact Parameter Arithmetic

* **Source Definition:** `CNN1D` in [`Approch-2-DL/scripts/submission_models.py`](../../Approch-2-DL/scripts/submission_models.py) L21–27.
* **Checkpoint / Artifact Size:** `cnn1d_best.pt` (`65,591` bytes).
* **Distinction:** **Highest Track A Test F1 (`90.91%`), Highest Track A Test MCC (`0.9071`), and Best Neural Performer Under Chronological Drift (`temporal AP = 0.9172`, `F1 = 0.8108`)**.

| Layer Stage | Verified Tensor Operation | Shape Trace | Exact Parameter Count |
| :---: | :--- | :---: | :---: |
| `Reshape` | `x.unsqueeze(1)` | `(B, 29)` → `(B, 1, 29)` | `0` |
| `Conv1` | `Conv1d(1, 8, kernel_size=3, padding=1)` → `ReLU` | `(B, 1, 29)` → `(B, 8, 29)` | `(1 × 8 × 3) + 8 = 32` |
| `Conv2` | `Conv1d(8, 16, kernel_size=3, padding=1)` → `ReLU` | `(B, 8, 29)` → `(B, 16, 29)` | `(8 × 16 × 3) + 16 = 400` |
| `Flatten` | `Flatten()` → `Dropout(0.2)` | `(B, 16, 29)` → `(B, 464)` | `0` |
| `FC1` | `Linear(464, 32)` → `ReLU` (`16 * 29 = 464`) | `(B, 464)` → `(B, 32)` | `464 × 32 + 32 = 14,880` |
| `Head` | `Linear(32, 1)` | `(B, 32)` → `(B, 1)` | `32 × 1 + 1 = 33` |
| **Sum** | **Two-Layer 1D Feature Convolution + Readout** | `(B, 29)` → `(B, 1)` | **`15,345` Trainable Parameters** |

### 4.2 Why `CNN1D` Works Empirically on `feature_names.txt` (And Its Critical Schema Fragility!)

Notice an important structural insight: tabular columns normally lack spatial invariance, so `1D` convolutions across columns often underperform pure MLPs. Here, however, `Approch-1/data/processed/feature_names.txt` places continuous predictors **contiguously by semantic family**:

* Sliding window at positions `[0..2]` captures `(ClaimAmount, PatientAge, PatientIncome)` simultaneously in a single `k=3` filter kernel!
* Sliding window `[3..5]` isolates `(ClaimYear, ClaimMonth, ClaimDayOfWeek)`!
* Sliding window `[5..7]` spans `(ClaimDayOfWeek, Cluster, PatientGender_F)`!

And notice where parameter capacity lives: `97.0%` (`14,880 / 15,345`) of weights sit in `Linear(464, 32)`, allowing `Conv1`+`Conv2` to behave like a bank of `16` shared multi-scale feature extractors before dense aggregation. **Warning:** permuting column order in `feature_names.txt` breaks this adjacency alignment completely and invalidates serialized `cnn1d_best.pt` weights!

### 4.3 Quantitative Evaluation Profile

* **Track A Validation / Convergence:** Best checkpoint **Epoch `41`** (early stop at Epoch `56`).
* **Track A Held-Out Test (`N = 675`, `40` Frauds):**
  * Accuracy: **`98.81%`** | Precision: `83.33%` | Recall: **`100.00%`** | F1: **`90.91%`** | ROC: `99.92%` | PR-AUC: `98.82%`
  * Batch Latency: `0.766 ms`
  * Confusion Counts: **`TN = 627`**, **`FP = 8`**, **`FN = 0`**, **`TP = 40`**
  * Derived Rates: Specificity `98.74%`, `FPR = 1.26%`, `FNR = 0.00%`, `FDR = 16.67%`, `NPV = 100.00%`, **`MCC = 0.9071`**.
* **Track B `temporal` Resilience:** Outperforms `ResidualMLP` (`+11.93` AP pts), `MLP` (`+5.91` AP pts), `SVM`, and `LogisticRegression`, holding **`0.9172` AP / `0.8108` F1** on chronological future data!

---

## 5. Model Card `CARD-DL-04`: Feature-Token `TabTransformer`

### 5.1 Card Identity & Multi-Head Attention Mechanics

* **Source Definition:** `TabTransformer` in [`Approch-2-DL/scripts/dl_advanced_models.py`](../../Approch-2-DL/scripts/dl_advanced_models.py) L16–57.
* **Checkpoint / Artifact Size:** `tabtransformer_best.pt` (`351,993` bytes — largest Track A neural weight file, `17.7×` `mlp_best.pt`).
* **Track B Status:** Not exported to Track B pure-JS browser runner (`export_browser.py` supports deterministic ML + compact `MLP`/`ResidualMLP`/`CNN1D` architectures only).

| Stage | Verified Transformer Pipeline Configuration (`dl_advanced_models.py`) | Tensor Shape Flow |
| :---: | :--- | :---: |
| `Tokenize` | Reshapes scalar features and applies shared `Linear(1, 32)` per feature coordinate | `(B, 29)` → `(B, 29, 1)` → `(B, 29, 32)` |
| `Encoder` | `TransformerEncoder(num_layers=2)` using `TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)` | `(B, 29, 32)` → `(B, 29, 32)` |
| `Head` | `Flatten()` (`29 × 32 = 928`) → `Linear(928, 64)` → `ReLU` → `Dropout(0.1)` → `Linear(64, 1)` | `(B, 29, 32)` → `(B, 928)` → `(B, 64)` → `(B, 1)` |

### 5.2 Quantitative Evaluation Profile & Latency Penalty

* **Convergence:** Stopped at best **Epoch `34`**.
* **Track A Held-Out Test (`N = 675`, `40` Frauds):**
  * Accuracy: `98.67%` | Precision: `81.63%` | Recall: **`100.00%`** | F1: `89.89%`
  * **Test ROC-AUC: `99.94%`** *(Highest in DL)* | **Test PR-AUC: `99.11%`** *(Highest in DL!)*
  * Confusion Counts: **`TN = 626`**, **`FP = 9`**, **`FN = 0`**, **`TP = 40`**
  * Derived Rates: Specificity `98.58%`, `FPR = 1.42%`, `FNR = 0.00%`, `FDR = 18.37%`, `NPV = 100.00%`, `MCC = 0.8971`.
* **Compute Bottleneck (`18.647 ms` Batch Latency):** Computing $29 \times 29 = 841$ attention weights across `4` heads × `2` layers costs **`18.647 ms`** total batch inference time — **`24.3×` slower than `CNN1D`** (`0.766 ms`) and **`27.7×` slower than `ResidualMLP`** (`0.674 ms`). Clearly demonstrates diminishing returns for attention mechanisms on small `d = 29` tabular inputs relative to gradient-boosted trees or `CNN1D`.

---

## 6. Model Card `CARD-DL-05`: Unsupervised `Autoencoder`

### 6.1 Card Identity, Verified Topology & Serving Transformation

* **Source Definition:** `Autoencoder` in [`Approch-2-DL/scripts/dl_advanced_models.py`](../../Approch-2-DL/scripts/dl_advanced_models.py) L60–116; trained inside [`train_advanced.py`](../../Approch-2-DL/scripts/train_advanced.py) L162–284.
* **Checkpoint / Artifact Size:** `autoencoder_best.pt` (`14,411` bytes).
* **Topology Correction (`DISC-01`):** Exact code dimensions are **`29 → 32 → 10 → 32 → 29`** (`Linear(29,32)+ReLU -> Linear(32,10)` Encoder; `Linear(10,32)+ReLU -> Linear(32,29)` Decoder), summing to **`2,599` parameters** — NOT `29 → 16 → 8 → 16 → 29` stated in older documentation.

| Component | Exact Layers (`dl_advanced_models.py`) | Shape Trace | Parameter Count |
| :--- | :--- | :---: | :---: |
| **Encoder** | `Linear(29, 32)` → `ReLU` → `Linear(32, 10)` | `(B, 29)` → `(B, 32)` → `(B, 10)` | `(960) + (330) = 1,290` |
| **Decoder** | `Linear(10, 32)` → `ReLU` → `Linear(32, 29)` | `(B, 10)` → `(B, 32)` → `(B, 29)` | `(352) + (957) = 1,309` |
| **Total** | Symmetric Bottleneck AE | `(B, 29)` → `(B, 29)` | **`2,599` Parameters** |

### 6.2 One-Class Objective, Serving Formula & Scientific Refutation Profile

* **Training Subset:** Exclusively benign training records (`y == 0`, `N = 2,961`), optimizing `MSELoss` with `Adam(lr=1e-3)` for `35` epochs.
* **Threshold Selection (`autoencoder_metrics.json`):** 95th percentile of validation reconstruction MSE across `N_val = 675`:
  $$\tau_{0.95} = 0.12858837842941284$$
* **Live Serving Calibration (`ml-service/app/services/dl_service.py`):** Maps error $e = \frac{1}{29}\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$ cleanly via `is_fraud = (e >= threshold)` and continuous uncalibrated score:
  $$s_{\text{AE}} = \frac{1}{1 + \exp\!\big(-5(e - \tau_{0.95}) / \max(\tau_{0.95}, 10^{-8})\big)}$$
* **Held-Out Test Failure Diagnostics (`N = 675`, `40` Frauds):**
  * Accuracy: `90.37%` | Precision: `3.70%` | Recall: `2.50%` | F1: `2.99%`
  * ROC-AUC: **`46.02%`** *(below coin toss `50%`)* | PR-AUC: **`5.23%`** *(below random prevalence `5.93%`)*
  * Batch Latency: **`0.498 ms`** *(fastest DL model, `2,599` params)*
  * Confusion Counts: **`TN = 609`**, **`FP = 26`**, **`FN = 39`**, **`TP = 1`**
  * Derived Rates: Specificity `95.91%`, `FPR = 4.09%`, **`FNR = 97.50%`**, **`FDR = 96.30%`**, `NPV = 93.98%`, **`MCC = -0.0192`**.
* **Out-of-Scope Verdict:** Unsuitable for fraud scoring on claims whose dollar envelope (`\$7,465.07–\$9,978.43`) resides wholly inside the continuous legitimate support (`\$100.12–\$9,997.20`), causing benign multi-categorical variance to dominate reconstruction MSE over subtle monetary+cluster frauds.
