# Approch-2-DL

## Purpose

This directory contains the Deep Learning component of Mini-Project 2: Health Insurance Fraud Detection. `Approch-1/` is the existing classical ML implementation owned by Ganesh and is read-only input for comparison.

## Current Status

The first DL milestone is complete. A PyTorch MLP has been implemented, trained, evaluated, checkpointed, and verified end-to-end.

Only the MLP is complete. The other DL models and website/backend integration are not implemented yet.

## Environment

Use the dedicated virtual environment for every project Python command:

```text
Approch-2-DL/.venv
```

Verified environment:

- Python: `3.13.1`
- PyTorch: `2.14.0+cpu`
- NumPy: `2.5.3`
- pandas: `3.0.6`
- scikit-learn: `1.9.1`
- matplotlib: `3.11.2`
- joblib: `1.6.0`
- openpyxl: `3.1.5`

Do not use global Python for this project.

## MLP Architecture

- Framework: PyTorch
- Input: 29 features
- Architecture: `29 → 64 → 32 → 1`
- Weighted binary cross-entropy with logits for class imbalance
- Validation PR-AUC for checkpoint selection
- Early stopping
- Seed: `42`
- Reported decision threshold: `0.5`

The MLP uses the frozen train/validation/test arrays from the existing Approach-1 leakage-excluded contract. The default split is train 3,150 / validation 675 / test 675, with fraud counts 189 / 41 / 40.

## Training

- Device: CPU
- Best epoch: `34`
- Total epochs completed: `49`
- Training time: approximately `4.430 seconds` in the verified `.venv` run

## Test Results

| Metric | Test result |
|---|---:|
| PR-AUC | 0.953751 |
| ROC-AUC | 0.997283 |
| Precision | 0.847826 |
| Recall | 0.975000 |
| F1 | 0.906977 |
| Accuracy | 0.988148 |

Confusion matrix at threshold 0.5:

```text
TN = 628, FP = 7
FN = 1,   TP = 39
```

## Artifacts

- `models/mlp/mlp_best.pt`
- `results/mlp_metrics.json`
- `results/mlp_metrics.csv`
- `results/inference_smoke_test.json`
- `evaluation/mlp_confusion_matrix.png`
- `evaluation/mlp_roc_curve.png`
- `evaluation/mlp_pr_curve.png`
- `evaluation/mlp_training_history.png`
- `logs/training_history.json`
- `logs/mlp_training_summary.json`

The frozen local feature snapshot is under `results/processed/`.

## How to Run

From the repository root:

```powershell
.\Approch-2-DL\.venv\Scripts\Activate.ps1

python Approch-2-DL\scripts\train_mlp.py
python Approch-2-DL\scripts\evaluate_mlp.py
```

The direct-interpreter form is:

```powershell
& '.\Approch-2-DL\.venv\Scripts\python.exe' 'Approch-2-DL\scripts\train_mlp.py'
& '.\Approch-2-DL\.venv\Scripts\python.exe' 'Approch-2-DL\scripts\evaluate_mlp.py'
```

The scripts resolve paths from their own location. A different processed contract can be supplied with `--source-processed-dir`.

## Methodological Notes

The current baseline uses the existing leakage-excluded 29-feature preprocessing contract. These results are a benchmark comparison, not proof of real-world fraud-detection performance.

- `ClaimStatus` is excluded because it is an apparent post-adjudication outcome and would leak information unavailable at claim intake.
- `Cluster` is retained for baseline comparability, but 267 of 270 fraud cases occur in Cluster 1. It may be a target proxy or engineered shortcut; ablation and temporal validation are required.
- `ClaimAmount` is retained, but its strong class separation may also be an engineered shortcut.
- `ClaimID`, `PatientID`, and `ProviderID` are unique in this workbook. `DiagnosisCode`, `ProcedureCode`, and `ProviderLocation` are near-unique and are excluded rather than blindly one-hot encoded.
- The current benchmark uses a random stratified split. A future-time temporal holdout is still required.

## Future DL Models

1. MLP — **COMPLETED**
2. Residual MLP — **PENDING**
3. 1D CNN — **PENDING**
4. TabTransformer — **PENDING**
5. Autoencoder — **PENDING**

## Integration Plan

The following is planned architecture only:

```text
React + TypeScript
	↓
Node.js + Express
	↓
Python + FastAPI
	↓
ML / DL / Agentic components
```

Website and backend integration are **NOT IMPLEMENTED YET**.
