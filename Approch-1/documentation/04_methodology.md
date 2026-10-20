# Training and evaluation methodology

[Documentation index](README.md)

## Models as implemented

[train_all_models.py](../scripts/train_all_models.py) fits the following estimators to the processed training matrix. Parameters omitted from this table retain their library defaults, so library versions affect reproducibility.

| Saved model key | Estimator | Explicit configuration |
|---|---|---|
| LogisticRegression | `LogisticRegression` | `max_iter=2000`, `class_weight='balanced'`, `solver='lbfgs'`, `random_state=42` |
| RandomForest | `RandomForestClassifier` | `n_estimators=300`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1` |
| XGBoost | `XGBClassifier` | `use_label_encoder=False`, `eval_metric='logloss'`, `scale_pos_weight=2961/189`, `random_state=42`, `n_jobs=-1` |
| SVM | `SVC` | `probability=True`, `class_weight='balanced'`, `kernel='rbf'`, `random_state=42` |
| GradientBoosting | `HistGradientBoostingClassifier` | `class_weight='balanced'`, `random_state=42` |

`GradientBoosting` is the artifact name for the histogram-based estimator. It is not `GradientBoostingClassifier`. XGBoost's positive-class weight is hard-coded to the saved training ratio, approximately 15.67; it must be reconsidered if the split or dataset changes.

The models provide a linear baseline, bagged trees, two boosting methods, and a nonlinear margin classifier. Weighting increases the contribution of the minority fraud class to training. The source contains no hyperparameter search or cross-validation loop.

## Training sequence

1. Load the saved train, validation, and test NPY arrays.
2. Fit each estimator on the training arrays.
3. Compute validation and test labels with `model.predict` and ranking scores with `predict_proba(... )[:, 1]`.
4. Record accuracy, precision, recall, F1, ROC AUC, and average precision for both partitions.
5. Save each fitted estimator and test confusion matrix, ROC curve, and precision-recall curve.
6. Write aggregated CSV/JSON metrics, select the largest `Val_F1`, and write the model key to `best_model.txt`.
7. Write fit time, model size, a 100-row prediction timing, software metadata, and split counts to the training log.

The selection expression uses pandas `idxmax`; an exact validation-F1 tie resolves to the first maximum in the table. The saved winner is GradientBoosting with validation F1 = 1.0000.

## Metric definitions

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of all labels predicted correctly |
| Precision | `TP / (TP + FP)`: fraction of flagged claims labeled Fraud |
| Recall | `TP / (TP + FN)`: fraction of labeled fraud detected |
| F1 | Harmonic mean of precision and recall |
| ROC AUC | Score ranking across false-positive and true-positive rates |
| `PR_AUC` in saved files | `average_precision_score`, a weighted precision summary over recall increments |
| FP / FN | Legitimate claims flagged / fraudulent claims missed |

The field called `PR_AUC` is average precision, not trapezoidal integration of the plotted PR curve. `precision_score`, `recall_score`, and `f1_score` use `zero_division=0`.

Label-based metrics come from each estimator's `predict` method. This matters for SVC: its probability estimates and decision-function classification are distinct, so a universal `predict_proba >= 0.5` rule is not an exact replacement for the reported labels.

## Evaluation discipline

Validation F1 is the implemented selection rule. Test metrics describe the held-out split and should not be used to choose thresholds or tune models. The scripts load and score the test arrays again when rerun; they do not enforce a one-time test lock. Comparison and error-analysis helpers also reuse test data.

There is no business-cost threshold optimization, calibration assessment, temporal holdout, confidence-interval analysis, or fairness audit implemented by these scripts.

## Reporting helpers

| Script | Behavior |
|---|---|
| `comparison_plots.py` | Compares validation/test F1 and test ROC AUC; plots available feature importances for RandomForest and XGBoost |
| `error_analysis.py` | Reloads all five models, predicts the test set, writes FP/FN/TP/TN to `result_analysis.txt`, and saves an error-count chart |
| `create_notebooks.py` | Produces five notebook templates with metric text copied from the CSV |

HistGradientBoosting does not expose the `feature_importances_` attribute used by the plotting helper. The helper does not generate an importance figure for that estimator. Notebook templates contain unexpanded placeholders and incomplete model imports; the training script is the supported reproduction path.

The Markdown error report is a maintained narrative; `error_analysis.py` regenerates its text counterpart, not the Markdown file.
