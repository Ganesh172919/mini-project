# KEY RESULTS — Medical Insurance Fraud Detection (Approach 1: Classic ML)

> Single source: `Health Insurance Fraud Claims.xlsx` (4,500 claims). Split 70/15/15 seed 42. Leakage column `ClaimStatus` excluded. All numbers below are from `evaluation/model_results.csv` (real run, never hand-typed). Test block = 675 claims (40 fraud), scored once per model.

## 1. At-a-glance

- **Best by Val F1 (selection):** GradientBoosting (1.0000)
- **Best Test F1:** XGBoost 0.9877 (ROC 1.0000, 1 FP / 0 FN)
- **Tie second:** RandomForest & GradientBoosting 0.9873 (0 FP / 1 FN each)
- **Baseline zero-miss option:** LogisticRegression recall 1.0000 at precision 0.625 (24 FP, 0 FN)
- **Control mid-tier:** SVM F1 0.8352 (13 FP, 2 FN)

## 2. Full table

| Model | Val Acc | Val Prec | Val Rec | Val F1 | Val ROC | Val PR | Test Acc | Test Prec | Test Rec | Test F1 | Test ROC | Test PR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.9778 | 0.7407 | 0.9756 | 0.8421 | 0.9945 | 0.9411 | 0.9644 | 0.6250 | 1.0000 | 0.7692 | 0.9950 | 0.9186 |
| RandomForest | 0.9985 | 1.0000 | 0.9756 | 0.9877 | 1.0000 | 1.0000 | 0.9985 | 1.0000 | 0.9750 | 0.9873 | 1.0000 | 0.9994 |
| XGBoost | 0.9985 | 1.0000 | 0.9756 | 0.9877 | 1.0000 | 0.9994 | 0.9985 | 0.9756 | 1.0000 | 0.9877 | 1.0000 | 1.0000 |
| SVM | 0.9837 | 0.8000 | 0.9756 | 0.8791 | 0.9941 | 0.9442 | 0.9778 | 0.7451 | 0.9500 | 0.8352 | 0.9955 | 0.9252 |
| GradientBoosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9985 | 1.0000 | 0.9750 | 0.9873 | 0.9847 | 0.9764 |

## 3. Error counts (money view)

| Model | FP (clean flagged) | FN (fraud paid) | Desk meaning |
|---|---:|---:|---|
| LogisticRegression | 24 | 0 | Miss nothing, investigate 24 extra |
| RandomForest | 0 | 1 | No false alarms, 1 fraud slips |
| XGBoost | 1 | 0 | Catch all, 1 false alarm |
| SVM | 13 | 2 | Middle |
| GradientBoosting | 0 | 1 | Same as RF (selected) |

Operational choice = which error is priced higher. Default threshold 0.5; cost curve is the proper selector.

## 4. Curves & matrices

- ROC panels: `evaluation/roc_{Model}.png` — tree ensembles ≥0.9999
- PR panels: `evaluation/pr_{Model}.png` — PR-AUC matters at 6% fraud; XGBoost 1.0, RF 0.9994
- Confusion: `evaluation/cm_{Model}.png`
- F1 comparison: `evaluation/model_comparison_f1.png`
- Error bars: `evaluation/error_analysis_fp_fn.png`

## 5. Feature importance (impurity, tree models)

- ClaimAmount > Cluster > ClaimMonth/Year. See `evaluation/feature_importance_*.png`.
- ClaimStatus excluded — post-adjudication leak. Impurity importance is hypothesis-generating, not causal; recommend permutation importance next.

## 6. Serving

- Flask `project/website/app.py` loads all 5 joblib on `:5000`; `/api/health` shows `ranked` + `busy`.
- Smart routing: `_pick_model(requested)` walks ranked Val_F1 list for first non-busy flag (`/tmp/busy_{model}.flag`), headers `X-Routed-From`/`X-Served-By`, `routed=true`.
- Audit: `website/prediction_log.txt` (timestamp|requested|served|pred|prob|latency|routed).

## 7. Siblings (same split contract)

- Approach 2 DL+XAI `:8010` — 5 deep nets + calibration + XAI faith/stability, composite leaderboard.
- Approach 3 Agentic `:8020` — 6 agents, ROC 0.9926, triage APPROVE/REVIEW/FRAUD, smart fallback to `:5000` if agent busy.
- Repro: `python project/scripts/*.py` (9 commands) + `00_run_all.py` per sibling, seed 42.

## 8. Limits to quote

- Random split, not temporal; `Cluster` dominates (single-cohort risk); 40 fraud in test (±2.5pp per claim); scores uncalibrated in this folder (use Approach 2 ECE).

*Generated from `project/evaluation/model_results.csv` @ 2026-09-16; PDF/PPT read this file at build time.*
