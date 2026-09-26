# Result Analysis — Approach 1 (Real Data Only)

## Method
- Stratified 70/15/15 split on 4,500 real claims.
- 5 algorithms evaluated on real validation (675) and test (675) sets.
- Leakage control: `ClaimStatus` (Approved/Denied/Pending, the insurer's post-adjudication outcome) is excluded from the feature registry; the frozen registry is 29 columns (7 numeric + 22 one-hot).
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
- Error analysis computed directly on test predictions (no synthetic injection).

## Real Results (Test Set — 675 claims, 40 fraud)

| Model | FP | FN | TP | TN | F1 | ROC-AUC | Notes |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 24 | 0 | 40 | 611 | 0.769 | 0.995 | High recall, lower precision (24 false alarms) |
| RandomForest | 0 | 1 | 39 | 635 | 0.987 | 1.000 | Near-perfect; 1 missed fraud |
| XGBoost | 1 | 0 | 40 | 634 | 0.988 | 1.000 | Excellent; 1 false alarm |
| SVM | 13 | 2 | 38 | 622 | 0.835 | 0.995 | Mid-tier; more errors |
| GradientBoosting | 0 | 1 | 39 | 635 | 0.987 | 0.985 | Tie best with RF |

## Key Observations
1. Tree ensembles (RandomForest, XGBoost, GradientBoosting) dominate structured tabular fraud data.
2. Near-perfect ROC-AUC (0.99–1.00) suggests very strong predictive signals — `Cluster` and `ClaimAmount` combinations — rather than pure randomness. This is not an error in calculation; it reflects the dataset's design (near-unique diagnosis/procedure codes were dropped to prevent overfit, and `ClaimStatus` was removed as post-adjudication leakage, but the engineered `Cluster` column remains highly informative).
3. Logistic Regression catches all fraud (FN=0) but raises 24 false positives — useful if false negatives are very costly (e.g., missing large fraud claims).
4. SVM provides a middle ground with moderate errors.
5. **Recommendation:** Before production deployment, validate on a true temporal holdout (post-2024 claims) and aggregate remaining high-cardinality clinical codes into CCS/DRG groups. Also tune decision thresholds using business cost of false positives vs missed fraud.

## Error Plot
- `project/evaluation/error_analysis_fp_fn.png` — visual comparison of false positives and false negatives per model.
- `project/evaluation/result_analysis.txt` — raw counts and recommendations.
