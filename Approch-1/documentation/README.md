# Approach 1 documentation

Updated: 20 October 2026.

This is the maintained documentation for the classical machine learning approach to medical insurance claim fraud detection. It describes the source, dataset, saved artifacts, and integration code present in this checkout. The older `docs/` entry points link here.

## Reading guide

| Document | Contents |
|---|---|
| [Project overview](01_project_overview.md) | Objective, scope, architecture, and repository layout |
| [Dataset and preprocessing](02_dataset_analysis.md) | Data audit, field dictionary, leakage exclusion, splits, and feature contract |
| [Exploratory analysis](03_eda.md) | Verified observations and the ten existing EDA figures |
| [Methodology](04_methodology.md) | Five estimators, training workflow, model selection, and evaluation rules |
| [Results](05_results.md) | Saved validation/test metrics, error counts, and interpretation |
| [Setup and usage](06_setup_and_usage.md) | Windows setup, legacy path compatibility, read-only inference, and troubleshooting |
| [Learning notes and next steps](07_learning_notes.md) | Limitations, reproducibility, integration boundaries, and maintenance checklist |

## Current status

Present: preprocessing and training scripts, five fitted models, processed splits, metrics, evaluation figures, and notebook templates. The saved validation-selected model is **GradientBoosting**; **XGBoost** has the highest saved test F1.

The `Approch-1/website/` and `Approch-1/outputs/` directories, Flask entry point, and PDF/PPT generation scripts described in earlier documentation are absent. The shared FastAPI service consumes the Approach 1 preprocessor but predicts with the Approach 2 MLP. It does not serve the five classical estimators.

## Evidence and provenance

- [Preprocessing source](../scripts/preprocess.py) and [training source](../scripts/train_all_models.py).
- [Metric CSV](../evaluation/model_results.csv), [metric JSON](../evaluation/model_results.json), and [selected model](../evaluation/best_model.txt).
- [Training log](../logs/training_metrics.json) and [error counts](../evaluation/result_analysis.txt).
- [Ordered feature names](../data/processed/feature_names.txt).

Dataset counts, category cardinalities, split shapes, and the matching Approach 2 feature-name list were checked against local files for this update. Tables reproduce saved results rounded to four decimals; this documentation update did not retrain models. The saved log records a run generated on 13 September 2026.

The documented inference example was run against the saved GradientBoosting model and returned a Legitimate prediction with 29 input features. Local Markdown links, result-table values, and PowerShell example syntax were also checked. The full retraining workflow was not rerun for this documentation update.
