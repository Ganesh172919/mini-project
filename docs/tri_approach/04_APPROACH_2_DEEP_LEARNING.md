# Volume 04 — Approach 2: Deep Learning & Tabular Representation Suite (`Approch-2-DL/`)

> **Document Classification:** Neural Architecture Specification, Training Dynamics & Empirical Evaluation  
> **Primary Ownership:** **M. Jagadeeswar (`23BDS033`)**  
> **Workspace Path:** [`Approch-2-DL/`](../../Approch-2-DL/) *(legacy spelling preserved under `ADR-01`)*  
> **Supervised Optimization Protocol:** `BCEWithLogitsLoss(pos_weight=2961/189)`, `AdamW(lr=1e-3, weight_decay=1e-4)`, `batch_size=128`, Early Stopping (`patience=15` monitoring `val_pr_auc`), `seed=42`.  
> **Primary Ground-Truth Sources:** [`Approch-2-DL/README.md`](../../Approch-2-DL/README.md), [`Approch-2-DL/scripts/dl_common.py`](../../Approch-2-DL/scripts/dl_common.py), [`submission_models.py`](../../Approch-2-DL/scripts/submission_models.py), [`dl_advanced_models.py`](../../Approch-2-DL/scripts/dl_advanced_models.py), [`train_mlp.py`](../../Approch-2-DL/scripts/train_mlp.py), [`train_supervised.py`](../../Approch-2-DL/scripts/train_supervised.py), [`train_advanced.py`](../../Approch-2-DL/scripts/train_advanced.py), [`evaluate_mlp.py`](../../Approch-2-DL/scripts/evaluate_mlp.py), [`Approch-2-DL/results/*_metrics.json`](../../Approch-2-DL/results/mlp_metrics.json), and [`Approch-2-DL/processed/contract_metadata.json`](../../Approch-2-DL/processed/contract_metadata.json).

---

## 1. Subsystem Philosophy & Upstream Data Contract Alignment

Approach 2 evaluates whether neural representation learning across **five distinct PyTorch inductive biases** can rival or complement axis-aligned tree ensembles on tabular health insurance claims without altering data splits, scaling parameters, or feature ordering.

### 1.1 Split-Seed & Tensor Parity Guarantee

To ensure differences between Approach 1 (`Approch-1/`) and Approach 2 (`Approch-2-DL/`) isolate **pure model inductive bias** rather than split noise:

1. [`Approch-2-DL/processed/`](../../Approch-2-DL/processed/) mirrors the exact six `.npy` arrays (`X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`) produced by `Approch-1/scripts/preprocess.py` (`3150 / 675 / 675`, `29` features).
2. [`contract_metadata.json`](../../Approch-2-DL/processed/contract_metadata.json) locks feature order, input dimensionality (`29`), preprocessing lineage, and positive/negative split statistics (`189` frauds / `2,961` legitimate in training).
3. All tensor loaders inside `dl_common.py` cast inputs to `torch.float32` and binary targets to column shape `(N, 1)` so `BCEWithLogitsLoss` avoids broadcasting shape bugs.

---

## 2. Directory Topology & Script Execution Architecture (`Approch-2-DL/`)

```text
Approch-2-DL/
|-- README.md                           # Subsystem guide & experiment overview (139 lines)
|-- requirements.txt                    # PyTorch + NumPy + scikit-learn dependencies (7 lines)
|-- processed/
|   |-- contract_metadata.json          # Feature contract & split statistics (49 lines)
|   |-- feature_names.txt               # Mirror of canonical 29-feature schema
|   `-- X_{train,val,test}.npy, y_*.npy # Identical arrays from Approch-1
|-- scripts/
|   |-- dl_common.py                    # DataLoaders, seed, MLP class, metrics & eval helpers (170 lines)
|   |-- submission_models.py            # ResidualMLP & CNN1D network definitions (30 lines)
|   |-- dl_advanced_models.py           # TabTransformer & Autoencoder definitions (116 lines)
|   |-- train_mlp.py                    # Standalone MLP trainer -> mlp_best.pt & history (153 lines)
|   |-- train_supervised.py             # Shared supervised runner -> ResidualMLP & CNN1D (173 lines)
|   |-- train_advanced.py               # Runner for TabTransformer + benign Autoencoder (302 lines)
|   `-- evaluate_mlp.py                 # Standalone post-hoc evaluation & smoke diagnostics (146 lines)
|-- models/
|   |-- mlp_best.pt                     # 19,873 bytes
|   |-- residualmlp_best.pt             # 80,161 bytes
|   |-- cnn1d_best.pt                   # 65,591 bytes
|   |-- tabtransformer_best.pt          # 351,993 bytes
|   `-- autoencoder_best.pt             # 14,411 bytes
|-- results/
|   |-- mlp_metrics.json                # 71 lines
|   |-- residualmlp_metrics.json        # 65 lines
|   |-- cnn1d_metrics.json              # 65 lines
|   |-- tabtransformer_metrics.json     # 65 lines
|   |-- autoencoder_metrics.json        # 68 lines
|   `-- inference_smoke_test.json       # Live batch inference smoke validation (9 lines)
`-- logs/
    |-- mlp_training_summary.json       # Best epoch & training summary for MLP (17 lines)
    `-- training_history.json           # Per-epoch loss, val_pr_auc, lr history (394 lines)
```

### 2.1 Shared Infrastructure Modules (`scripts/*.py`)

| Module | Primary Responsibility | Key Exported Symbols / Interfaces |
| :--- | :--- | :--- |
| **`dl_common.py`** (`170` lines) | Deterministic seeding (`seed=42`), numpy `.npy` ingestion, `TensorDataset`/`DataLoader(batch_size=128)`, `MLP` definition, unified binary metric computation (`Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`, `PR-AUC`, `CM`), and batched latency measurement. | `MLP`, data loader factory, metric/CM evaluator, checkpoint serializer |
| **`submission_models.py`** (`30` lines) | Defines compact tabular inductive modules reused across both Track A (`train_supervised.py`) and Track B (`submission/scripts/run_pipeline.py`). | `ResidualBlock`, `ResidualMLP`, `CNN1D` |
| **`dl_advanced_models.py`** (`116` lines) | Implements higher-capacity / unsupervised tabular architectures (`TabTransformer`, `Autoencoder`) tested exclusively in Track A. | `TabTransformer`, `Autoencoder` |
| **`train_mlp.py`** (`153` lines) | Dedicated training harness for baseline `MLP`; logs full epoch-by-epoch diagnostics into `logs/training_history.json` (`394` lines) and `logs/mlp_training_summary.json`. | `AdamW(lr=1e-3, weight_decay=1e-4)`, `patience=15` monitoring `val_pr_auc` |
| **`train_supervised.py`** (`173` lines) | Generalized supervised trainer executing `ResidualMLP` and `CNN1D` under identical `BCEWithLogitsLoss(pos_weight=2961/189)` + early stopping rules. | Produces `residualmlp_best.pt`, `cnn1d_best.pt`, and `*_metrics.json` |
| **`train_advanced.py`** (`302` lines) | Bifurcated trainer: runs supervised `TabTransformer` alongside unsupervised one-class `Autoencoder` (`y == 0` filter -> `MSELoss` -> 95th percentile validation threshold computation). | Produces `tabtransformer_best.pt`, `autoencoder_best.pt`, and `*_metrics.json` |
| **`evaluate_mlp.py`** (`146` lines) | Loads checkpoints from `models/`, verifies tensor evaluation & inference speed, and records `results/inference_smoke_test.json`. | Post-training checkpoint validator & smoke reporter |

---

## 3. Architectural Blueprint & Parameter Analysis of All Five Neural Models

A critical discipline of this documentation suite is documenting **only exact layer structures verified directly inside source files** (`dl_common.py`, `submission_models.py`, `dl_advanced_models.py`). Where earlier summaries differed (for example claiming `BatchNorm1d` in `MLP` or `29->16->8` layers in `Autoencoder`), the exact code reality is established below.

```text
+---------------------------------------------------------------------------------------------------+
|                           FIVE PYTORCH ARCHITECTURES IN Approch-2-DL/                             |
+---------------------------------------------------------------------------------------------------+

  (1) MLP [dl_common.py L50-64] — 4,033 Params | NO BatchNorm
      x(B,29) -> Linear(29,64)+ReLU+Drop(0.2) -> Linear(64,32)+ReLU+Drop(0.2) -> Linear(32,1) -> logit

  (2) ResidualMLP [submission_models.py L5-19] — 18,625 Params
      x(B,29) -> Linear(29,64)+ReLU
                 |
                 +-> ResBlock1: [Lin(64,64)+ReLU+Drop(0.2)+Lin(64,64)] (+) x -> ReLU
                 +-> ResBlock2: [Lin(64,64)+ReLU+Drop(0.2)+Lin(64,64)] (+) x -> ReLU
                 +-> Linear(64,1) -> logit

  (3) CNN1D [submission_models.py L21-27] — 15,345 Params
      x(B,29) -> unsqueeze(1): (B,1,29)
                 -> Conv1d(1,8,k=3,p=1)+ReLU -> Conv1d(8,16,k=3,p=1)+ReLU -> (B,16,29)
                 -> Flatten -> (B,464) -> Drop(0.2) -> Linear(464,32)+ReLU -> Linear(32,1) -> logit

  (4) TabTransformer [dl_advanced_models.py L16-57] — Column Self-Attention
      x(B,29) -> unsqueeze(-1): (B,29,1) -> Shared Linear(1,32): (B,29,32) [29 Feature Tokens]
                 -> TransformerEncoder(2x EncoderLayer: d_model=32, nhead=4, d_ff=128, drop=0.1)
                 -> Flatten -> (B,928) -> Linear(928,64)+ReLU+Drop(0.1) -> Linear(64,1) -> logit

  (5) Autoencoder [dl_advanced_models.py L60-116] — Unsupervised Benign Reconstruction (y == 0)
      x(B,29) -> Encoder: Linear(29,32)+ReLU -> Linear(32,10) [Bottleneck R^10]
              -> Decoder: Linear(10,32)+ReLU -> Linear(32,29) -> x_hat(B,29) -> MSE(x, x_hat)
```

### 3.1 Architecture 1: `MLP` (`dl_common.py` L50–64)

* **Exact Layer Sequence & Shape Trace (for input batch size $B$):**
  1. `Linear(29, 64)` → `(B, 64)` (`29 × 64 + 64 = 1,920` params) → `ReLU` → `Dropout(p=0.2)`
  2. `Linear(64, 32)` → `(B, 32)` (`64 × 32 + 32 = 2,080` params) → `ReLU` → `Dropout(p=0.2)`
  3. `Linear(32, 1)` → `(B, 1)` (`32 × 1 + 1 = 33` params)
* **Exact Trainable Parameter Total:** **`4,033` parameters** (`19,873` bytes serialized in `mlp_best.pt`).
* **Verified Design Finding (No `BatchNorm1d`):** Inspection of `dl_common.py` L50–64 confirms zero `BatchNorm1d` layers exist in `MLP`. Input normalization is handled solely upstream by Approach 1's `StandardScaler` (`0`-mean, unit-variance numeric cols + `{0, 1}` OHE indicators), meaning mini-batch activation distributions stay bounded cleanly without batch-statistic noise.
* **Inductive Behavior:** Provides `29 → 64` slight expansion before tapering (`64 → 32 → 1`). Converged at **Epoch `34`** (stopped after `15` epochs without improvement at Epoch `49`, taking **`4.43s`** total).

### 3.2 Architecture 2: `ResidualMLP` (`submission_models.py` L5–19)

* **Exact Layer Sequence & Shape Trace:**
  1. **Stem:** `Linear(29, 64)` → `ReLU` → `(B, 64)` (`1,920` params)
  2. **Residual Block 1 (`ResidualBlock`):**
     $$\mathbf{h}_1 = \operatorname{ReLU}\Big(\mathbf{h}_0 + \operatorname{Linear}_{64\to 64}\big(\operatorname{Dropout}_{0.2}(\operatorname{ReLU}(\operatorname{Linear}_{64\to 64}(\mathbf{h}_0)))\big)\Big)$$
     (`2 × (64 × 64 + 64) = 8,320` params)
  3. **Residual Block 2 (`ResidualBlock`):** Identical topology applied to $\mathbf{h}_1$ → `(B, 64)` (`8,320` params)
  4. **Classification Head:** `Linear(64, 1)` → `(B, 1)` (`65` params)
* **Exact Trainable Parameter Total:** **`18,625` parameters** (`80,161` bytes serialized in `residualmlp_best.pt`, `4.62×` `MLP`'s parameter count).
* **Inductive Behavior:** Identity skip connections preserve clean gradient flow and unaltered linear combinations of the `64`-dim stem embeddings alongside non-linear corrections. In Track A (`train_supervised.py`), best `val_pr_auc` peaked early at **Epoch `20`** (`100.00%` Test Recall, `95.61%` PR-AUC, `0.674 ms` total inference latency).
* **Determinism Trajectory Divergence (`Track A` vs. `Track B`):** Under Track B's strict `torch.use_deterministic_algorithms(True)` setting (`submission/results/baseline_ResidualMLP_training.json`), optimization followed a longer deterministic trajectory peaking at **Epoch `42`** (`4.42s`, Test PR-AUC `95.67%`).

### 3.3 Architecture 3: `CNN1D` (`submission_models.py` L21–27)

* **Exact Layer Sequence & Shape Trace:**
  1. **Sequence Reshape:** `x.unsqueeze(1)` converts `(B, 29)` into a single-channel 1D signal `(B, 1, 29)`.
  2. **Conv Stage 1:** `Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)` → `ReLU` → `(B, 8, 29)` (`1 × 8 × 3 + 8 = 32` params).
  3. **Conv Stage 2:** `Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)` → `ReLU` → `(B, 16, 29)` (`8 × 16 × 3 + 16 = 400` params).
  4. **Flatten & Dense Readout:** `Flatten()` → `(B, 464)` (`16 × 29`) → `Dropout(0.2)` → `Linear(464, 32)` (`14,880` params) → `ReLU` → `Linear(32, 1)` (`33` params).
* **Exact Trainable Parameter Total:** **`15,345` parameters** (`65,591` bytes serialized in `cnn1d_best.pt`). Note that `97.0%` (`14,880 / 15,345`) of parameters reside inside the flattened convolution-to-dense readout layer (`Linear(464, 32)`).
* **Why Column Order Matters Critically for `CNN1D`:** Applying `kernel_size=3, padding=1` slides 3-feature local filters across adjacent columns in `feature_names.txt`: indices `[0, 1, 2]` convolve `(ClaimAmount, PatientAge, PatientIncome)` simultaneously, while `[3, 4, 5]` convolve `(ClaimYear, ClaimMonth, ClaimDayOfWeek)` together! That accidental semantic grouping in `feature_names.txt` explains why `CNN1D` outperformed `MLP` and `ResidualMLP` significantly in Track A Test F1 (**`90.91%`**) and Test PR-AUC (**`98.82%`**, converged Epoch `41 / 56`). Any future reordering of `feature_names.txt` alters these local receptive neighborhoods and invalidates `cnn1d_best.pt`.

### 3.4 Architecture 4: `TabTransformer` (`dl_advanced_models.py` L16–57)

* **Exact Layer Sequence & Shape Trace:**
  1. **Scalar Feature Tokenization:** Reshapes `(B, 29)` → `(B, 29, 1)` and passes every scalar coordinate independently through a shared linear projection `Linear(1, 32)` → **`29` tokens of embedding width `d_model = 32`**, shape `(B, 29, 32)`.
  2. **Contextual Self-Attention Stack:** `TransformerEncoder(..., num_layers=2)` composed of `TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)`. Across each of `2` encoder layers, `4` attention heads (`d_k = 8` per head) compute global pairwise feature interactions across all `29 × 29 = 841` token pairs (`ClaimAmount` attending directly to `Cluster`, `ClaimType_Outpatient`, `PatientAge`, etc.) followed by residual LayerNorm and `32 → 128 → 32` position-wise FFN.
  3. **Flattened Readout:** Concatenates all `29` contextualized embeddings → `(B, 928)` (`29 × 32`) → `Linear(928, 64)` → `ReLU` → `Dropout(0.1)` → `Linear(64, 1)` → `(B, 1)`.
* **Operational Tradeoff Profile:**
  * Converged at **Epoch `34`**, serializes at **`351,993` bytes (`343.7 KB`)** (`17.7×` larger than `mlp_best.pt`).
  * Achieves **highest Track A Deep Learning Test ROC-AUC (`99.94%`) and Test PR-AUC (`99.11%`)**, matching `XGBoost`'s global ranking closely (`0` FN, `9` FP, `100.00%` Recall).
  * **Latency Penalty:** Computing quadratic attention matrices across `29` tokens × `2` layers yields **`18.647 ms` total batch inference time** (`~24×` slower than `CNN1D` `0.766 ms` and `~27×` slower than `ResidualMLP` `0.674 ms`).

### 3.5 Architecture 5: `Autoencoder` (`dl_advanced_models.py` L60–116 & `train_advanced.py` L162–284)

* **Verified Layer Sequence & Shape Trace:**
  1. **Encoder (`29 → 32 → 10`):**
     * `Linear(29, 32)` → `ReLU` (`29 × 32 + 32 = 960` params)
     * `Linear(32, 10)` (`32 × 10 + 10 = 330` params) — maps claim to a compressed `10`-dimensional latent code $\mathbf{z} \in \mathbb{R}^{10}$.
  2. **Decoder (`10 → 32 → 29`):**
     * `Linear(10, 32)` → `ReLU` (`10 × 32 + 32 = 352` params)
     * `Linear(32, 29)` (`32 × 29 + 29 = 957` params) — reconstructs full standardized continuous & binary vector $\hat{\mathbf{x}} \in \mathbb{R}^{29}$.
* **Total Trainable Parameters:** **`2,599` parameters** (`14,411` bytes serialized in `autoencoder_best.pt`).
* **Documentation Correction Notice:** Previous project overviews (`docs/02`, `docs/03`) inaccurately described this network as `29 → 16 → 8 → 16 → 29`. Direct source inspection of `Approch-2-DL/scripts/dl_advanced_models.py` L60–116 proves its true topology is **`29 → 32 → 10 → 32 → 29`**.
* **One-Class Training & Threshold Rule:** Trained for `35` epochs strictly on legitimate training records (`y == 0`, `N = 2,961`) using standard `Adam(lr=1e-3)` and unweighted `MSELoss`. Post-training, validation MSE scores across all `675` validation claims determine the 95th-percentile alarm boundary:
  $$\tau_{0.95} = 0.12858837842941284$$

---

## 4. Supervised Optimization & Convergence Dynamics

All four supervised architectures (`MLP`, `ResidualMLP`, `CNN1D`, `TabTransformer`) share strict training controls implemented across `dl_common.py`, `train_mlp.py`, `train_supervised.py`, and `train_advanced.py`:

| Control Hyperparameter | Exact Value Across Track A | Engineering Purpose |
| :--- | :--- | :--- |
| **Loss Function** | `BCEWithLogitsLoss(pos_weight=torch.tensor([2961 / 189]))` | Compensates `15.6667:1` majority majority skew inside binary gradient updates |
| **Optimizer** | `torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)` | Decoupled L2 weight decay mitigates overfitting on `N=3,150` tabular rows |
| **Batch Size** | `128` (`DataLoader(..., shuffle=True)` on train) | Yields `⌈3150 / 128⌉ = 25` mini-batch optimizer steps per epoch (`~1–2` frauds/batch avg if unweighted, `~7–8` scaled) |
| **Early Stopping Monitor** | `val_pr_auc` (Validation Average Precision), `patience = 15` | Prevents overfitting majority negative confidence and restores exact peak `*_best.pt` weights |
| **PRNG Seed** | `42` (Python, NumPy, PyTorch CPU/CUDA) | Standardizes weight initialization and batch shuffle sequencing across architectures |

### 4.1 Observed Convergence Profiles

* **`MLP` (`logs/training_history.json` & `mlp_training_summary.json`):** Ran `49` total epochs (`patience=15` triggered `15` epochs after best validation checkpoint at **Epoch `34`**) in **`4.43s`** total wall-clock time (`~90 ms/epoch`), demonstrating sub-5-second training agility on tabular arrays.
* **`ResidualMLP` (`residualmlp_metrics.json`):** Reached peak validation PR-AUC fastest at **Epoch `20`** in Track A (due to identity skip paths stabilizing early gradient signal), compared against **Epoch `42`** (`4.42s`) under Track B deterministic algorithms (`baseline_ResidualMLP_training.json`).
* **`CNN1D` (`cnn1d_metrics.json`):** Required slightly longer spatial motif convergence (**Epoch `41`** best checkpoint / `56` total epochs), rewarding patience with top neural Test F1 (**`90.91%`**).
* **`TabTransformer` (`tabtransformer_metrics.json`):** Peaked at **Epoch `34`** (matching `MLP`'s best epoch index), producing the highest neural continuous ranking (`99.11%` Test PR-AUC, `99.94%` ROC-AUC).

---

## 5. Comprehensive Empirical Results & Surgical Confusion-Matrix Audit

### 5.1 Master Track A Test Leaderboard (`N = 675`, `40` Frauds, `635` Legitimate)

| Model Architecture | Best Epoch | Checkpoint Bytes | Infer Batch Latency | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MLP`** | `34` *(stop `49`)* | `19,873 B` | `0.732 ms` | **`98.81%`** | **`84.78%`** | `97.50%` | `90.70%` | `99.73%` | `95.38%` |
| **`ResidualMLP`** | `20` | `80,161 B` | `0.674 ms` | `98.52%` | `80.00%` | **`100.00%`** | `88.89%` | `99.75%` | `95.61%` |
| **`CNN1D`** | `41` *(stop `56`)* | `65,591 B` | `0.766 ms` | **`98.81%`** | `83.33%` | **`100.00%`** | **`90.91%`** | `99.92%` | `98.82%` |
| **`TabTransformer`** | `34` | `351,993 B` | `18.647 ms` | `98.67%` | `81.63%` | **`100.00%`** | `89.89%` | **`99.94%`** | **`99.11%`** |
| **`Autoencoder`** | `35` *(unsup)* | `14,411 B` | **`0.498 ms`** | `90.37%` | `3.70%` | `2.50%` | `2.99%` | `46.02%` | `5.23%` |

### 5.2 Complete Confusion-Matrix Counts & Derived Operating Diagnostics

All operating statistics below are derived directly via exact arithmetic from the observed `(TN, FP, FN, TP)` quadruples persisted in `Approch-2-DL/results/*_metrics.json`:

| Model | `TN` | `FP` | `FN` | `TP` | Total Errors | Specificity (`TNR`) | False Pos Rate (`FPR`) | False Neg Rate (`FNR`) | False Disc Rate (`FDR`) | Neg Pred Val (`NPV`) | Matthews `MCC` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MLP`** | **`628`** | **`7`** | `1` | `39` | **`8`** | **`98.90%`** | **`1.10%`** | `2.50%` | **`15.22%`** | `99.84%` | `0.9032` |
| **`ResidualMLP`** | `625` | `10` | **`0`** | **`40`** | `10` | `98.43%` | `1.57%` | **`0.00%`** | `20.00%` | **`100.00%`** | `0.8874` |
| **`CNN1D`** | `627` | `8` | **`0`** | **`40`** | **`8`** | `98.74%` | `1.26%` | **`0.00%`** | `16.67%` | **`100.00%`** | **`0.9071`** |
| **`TabTransformer`** | `626` | `9` | **`0`** | **`40`** | `9` | `98.58%` | `1.42%` | **`0.00%`** | `18.37%` | **`100.00%`** | `0.8971` |
| **`Autoencoder`** | `609` | `26` | `39` | `1` | `65` | `95.91%` | `4.09%` | `97.50%` | `96.30%` | `93.98%` | **`-0.0192`** |

#### Key Quantitative Insights from Deep Learning vs. Classical Tree Comparisons

1. **Unbeatable Minority Recall Across Supervised Neural Nets:** Three of four supervised neural networks (`ResidualMLP`, `CNN1D`, `TabTransformer`) catch **`40 / 40` (`100.00%`)** frauds on Test, and even `MLP` catches `39 / 40`. Notice why: `BCEWithLogitsLoss(pos_weight=15.6667)` shifts the fixed `0.5` probability threshold (`z = 0`) downward onto true posterior probability $\approx 0.06$, casting a highly protective net!
2. **Threshold Shift Explains the Precision Gap vs. Trees (`80%–84.78%` vs. `97.56%–100%`):** Notice that `TabTransformer` reaches `99.94%` ROC-AUC and `99.11%` PR-AUC, proving its underlying *score ordering* separates classes almost cleanly — yet evaluating at an unadjusted logit threshold `z >= 0` (`p >= 0.5`) catches `9` legitimate borderline high-amount records (`FP = 9`), clipping F1 down to `89.89%`. Raising the threshold strictly via validation F1 optimization recovers precision (demonstrated directly in Track B's `submission/` experiments in Volume 07!).
3. **Architecture Tradeoff Summary Among Supervised DL Models:**
   * Best **F1 & Accuracy & MCC** + fast inference: **`CNN1D`** (`90.91%` F1, `MCC = 0.9071`, `0.766 ms`).
   * Best **Precision & Fewest False Positives** among DL: **`MLP`** (`84.78%` Prec, `FP = 7`, tiny `19.9 KB` file).
   * Best **Continuous Ranking (ROC/PR-AUC)**: **`TabTransformer`** (`99.94%` ROC, `99.11%` PR), at the cost of `24×` runtime latency (`18.647 ms`) and `17.7×` checkpoint bytes.

---

## 6. Scientific Post-Mortem: Why `Autoencoder` Fails (`PR-AUC = 5.23%`, `MCC = -0.0192`)

Among all empirical results across the entire repository, the `Autoencoder` outcome (`TN=609, FP=26, FN=39, TP=1`, `Precision = 3.70%`, `Recall = 2.50%`, `ROC-AUC = 46.02%`, `PR-AUC = 5.23%`, `MCC = -0.0192`) constitutes a vital scientific finding rather than an implementation error:

### 6.1 Three Geometric Causes of Failure

1. **In-Manifold Overlap Along `ClaimAmount` (`[\$7,465.07, \$9,978.43]`):** As proved in Volume 02 (`dataset_audit.json`), legitimate claims span `\$100.12` to `\$9,997.20` (`mean = \$4,776.80`). Roughly one quarter of the `2,961` legitimate training rows used to train `Autoencoder` already lie inside `\$7,465–\$9,997`. Therefore, when an actual fraud claim arrives with `ClaimAmount = \$8,733`, its feature vector falls squarely **inside the convex hull of normal training data**, producing low MSE well below $\tau_{0.95} = 0.1286$ (`39 / 40` frauds slip underneath the threshold!).
2. **Mixed Continuous-vs-One-Hot L2 Loss Distortion:** Standardized numerical features (`7` dimensions, variance $\approx 1.0$) coexist inside an unweighted `MSELoss` against `22` sparse $\{0, 1\}$ categorical indicators (`sum_c (x_c - x_hat_c)^2`). Rare legitimate combinations across specialties (`Cardiology`, `Neurology`), marital statuses, and submission channels consume decoder variance inside the `10`-dim bottleneck more strongly than the modest `1.3σ` `ClaimAmount` shift of a fraudulent claim — generating `26` false alarms on benign multi-categorical outliers while reconstructing frauds accurately!
3. **Sub-Chance Rank Correlation (`ROC-AUC = 46.02%`, `MCC = -0.0192`):** Because legitimate extreme values (e.g., very young/old claimants or low-probability categorical combinations) sit closer to the polytope corners than mid-8000s frauds in `Cluster == 1`, reconstruction error actually ranks slightly **anti-correlated** with fraud ground truth (`1 / 27` flagged precision vs `6.00%` random prior!).

### 6.2 Production Serving Impact (`ml-service/`)

Even though `Autoencoder` performs poorly offline, [`ml-service/app/services/dl_service.py`](../../ml-service/app/services/dl_service.py) preserves full serving support for educational comparison on `DLPage`:

* Loads `autoencoder_best.pt` and reads `threshold = 0.12858837842941284` directly from `Approch-2-DL/results/autoencoder_metrics.json`.
* Computes $e(x) = \text{mean}((x - \hat{x})^2)$, sets discrete `is_fraud = (e(x) >= threshold)`, and emits smooth sigmoid-scaled anomaly score $s_{\text{AE}} = \sigma(5 (e(x) - \tau) / \max(\tau, 10^{-8}))$ clearly marked as an uncalibrated anomaly score (`score_is_calibrated = False`).

---

## 7. Smoke Verification & Serving Handoff (`ml-service/`)

* **Offline Batch Verification (`results/inference_smoke_test.json`):** Recorded cleanly by `scripts/evaluate_mlp.py`, confirming model checkpoint loadability and deterministic batch outputs across test slices prior to FastAPI integration.
* **Live Serving Registry (`ml-service/app/services/dl_service.py` & `mlp_service.py`):** Exposes canonical keys `"mlp"`, `"residualmlp"`, `"cnn1d"`, `"tabtransformer"`, and `"autoencoder"` via `POST /predict/dl` and `POST /predict` (with `mlp_service.py` providing modular MLP loading/inference utilities).
* **Cross-Track Determinism Verification:** Covered extensively in Volume 07 (`submission/`), where `MLP`, `ResidualMLP`, and `CNN1D` are re-trained under strict `torch.use_deterministic_algorithms(True)` across `4` controlled scenarios (`baseline`, `intake`, `no_amount`, `temporal`) and exported directly to pure browser JavaScript arrays with maximum inference error `4.42e-7`!

---

## 8. Operational Runbook (Re-Running Approach 2 Experiments)

```powershell
# 1. Verify prerequisite Approach 1 arrays exist (or copy into processed/)
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\Approch-2-DL

# 2. Setup virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Train baseline MLP (outputs models/mlp_best.pt, logs/training_history.json, results/mlp_metrics.json)
python scripts/train_mlp.py

# 4. Train ResidualMLP & CNN1D (outputs residualmlp_best.pt, cnn1d_best.pt, *_metrics.json)
python scripts/train_supervised.py

# 5. Train TabTransformer & Autoencoder (outputs tabtransformer_best.pt, autoencoder_best.pt, *_metrics.json)
python scripts/train_advanced.py

# 6. Run evaluation smoke suite (outputs results/inference_smoke_test.json)
python scripts/evaluate_mlp.py
```
