# Key results: Approach 1 classical ML

Updated: 20 October 2026. This is a snapshot of the saved evaluation CSV, rounded to four decimals; no models were retrained for this documentation update.

- Dataset: 4,500 claims; 270 Fraud and 4,230 Legitimate.
- Stratified train/validation/test: 3,150 / 675 / 675, seed 42.
- Input contract: 29 transformed features; `ClaimStatus` excluded.
- Selected by validation F1: **GradientBoosting (1.0000)**.
- Highest saved test F1: **XGBoost (0.9877)**.

| Model | Val F1 | Test accuracy | Precision | Recall | Test F1 | ROC AUC | Average precision | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.8421 | 0.9644 | 0.6250 | 1.0000 | 0.7692 | 0.9950 | 0.9186 | 24 | 0 |
| RandomForest | 0.9877 | 0.9985 | 1.0000 | 0.9750 | 0.9873 | 1.0000 | 0.9994 | 0 | 1 |
| XGBoost | 0.9877 | 0.9985 | 0.9756 | 1.0000 | 0.9877 | 1.0000 | 1.0000 | 1 | 0 |
| SVM | 0.8791 | 0.9778 | 0.7451 | 0.9500 | 0.8352 | 0.9955 | 0.9252 | 13 | 2 |
| GradientBoosting | 1.0000 | 0.9985 | 1.0000 | 0.9750 | 0.9873 | 0.9847 | 0.9765 | 0 | 1 |

Test contains 675 claims, including 40 frauds. One additional missed fraud changes test recall by 2.5 percentage points. `PR_AUC` in the saved files means average precision. The RandomForest ROC AUC rounds to 1.0000; its stored value is approximately 0.99996063.

Labels use each estimator's `predict` method, which is not interchangeable with a universal probability threshold for SVM. Scores have no independent calibration audit here. Global importance plots exist for RandomForest and XGBoost.

Approach 1 has no Flask website or report-generation scripts in this checkout. The shared FastAPI service uses this preprocessor for Approach 2 MLP inference.

[Full results and run metadata](documentation/05_results.md) | [Documentation](documentation/README.md) | [Metric CSV](evaluation/model_results.csv) | [Error counts](evaluation/result_analysis.txt)
