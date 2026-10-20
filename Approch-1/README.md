# Approach 1: Classical machine learning for insurance claim fraud detection

Updated: 20 October 2026.

Five classical classifiers use the supplied medical insurance claims workbook to distinguish `Fraud` from `Legitimate`. This folder contains the offline preprocessing/training workflow and saved evaluation artifacts.

**[Read the maintained documentation](documentation/README.md).**

## Documentation

- [Project overview and architecture](documentation/01_project_overview.md)
- [Dataset, preprocessing, and the 29-feature contract](documentation/02_dataset_analysis.md)
- [Exploratory data analysis](documentation/03_eda.md)
- [Training and evaluation methodology](documentation/04_methodology.md)
- [Saved results and error analysis](documentation/05_results.md)
- [Setup, reproduction, and local inference](documentation/06_setup_and_usage.md)
- [Limitations, learning notes, and next steps](documentation/07_learning_notes.md)

## Verified baseline

| Item | Saved or verified value |
|---|---|
| Workbook | `data/raw/Health Insurance Fraud Claims.xlsx` (matches the repository-root copy) |
| Raw data | 4,500 rows, 19 columns; 270 Fraud and 4,230 Legitimate |
| Split | Train 3,150 / validation 675 / test 675; stratified, seed 42 |
| Features | 13 inputs after date expansion; 29 transformed columns |
| Leakage exclusion | `ClaimStatus` is excluded from model inputs |
| Models | LogisticRegression, RandomForest, XGBoost, SVM, GradientBoosting |
| Validation-selected model | GradientBoosting, validation F1 = 1.0000 |
| Highest saved test F1 | XGBoost, F1 = 0.9877; 1 false positive and 0 false negatives |

Metrics come from [model_results.csv](evaluation/model_results.csv); selection comes from [best_model.txt](evaluation/best_model.txt). Test metrics describe 675 claims, including only 40 fraud cases. Scores have no independent calibration audit in this approach.

## Read the saved results

From the repository root in PowerShell:

```powershell
Get-Content '.\Approch-1\evaluation\best_model.txt'
Import-Csv '.\Approch-1\evaluation\model_results.csv' |
    Format-Table Model, Val_F1, Test_F1, Test_ROC_AUC, Test_PR_AUC
```

For training, first follow the [setup guide](documentation/06_setup_and_usage.md). The scripts still use legacy `project/` relative paths; the guide explains the required path mapping and which artifacts are overwritten. It also provides an inference example using the saved transformer and selected classical model without rebuilding artifacts.

## What is present

- Six scripts under `scripts/`, five fitted models under `models/`, and processed splits under `data/processed/`.
- Metrics, confusion matrices, ROC/PR curves, comparison figures, error counts, and ten EDA figures under `evaluation/`.
- Saved training metadata under `logs/` and five notebook templates under `notebooks/`.
- Maintained documentation under `documentation/`; older `docs/` entry points link to it.

The notebook templates contain placeholders and incomplete imports; use the training script as the reproduction entry point.

## Application status

`Approch-1/website/`, `Approch-1/outputs/`, and the PDF/PPT generation scripts mentioned in older material are absent from this checkout. No Approach 1 Flask service is available here. The shared [ML service](../ml-service/README.md) reuses this folder's fitted preprocessor but runs the Approach 2 MLP, rather than the classical models.

Other approaches live in `../Approch-2-DL/` and `../Approach-3-Agentic-AI/`. Architecture illustrations in `docs/images/` include design context; they do not establish that all services shown are operational.

## Project team

IIIT Dharwad, B.Tech (Data Science and AI). Faculty adviser: Prof. Ramesh Athe. Team: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024). Approach 1 owner recorded by the repository: Ganesh.
