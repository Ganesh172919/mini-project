# Learning notes, limitations, and maintenance

[Documentation index](README.md)

## Lessons supported by the current artifacts

**Use metrics that expose minority-class errors.** An all-Legitimate rule reaches 94% accuracy on the workbook. Precision, recall, F1, average precision, and confusion counts explain the tradeoff more clearly. For example, LogisticRegression and XGBoost both detect all 40 test frauds, but generate 24 and 1 false alarms respectively.

**Separate selection from reporting.** The implementation chooses GradientBoosting by validation F1. Reporting XGBoost's stronger test F1 does not make it the validation-selected winner. Tune any business threshold using validation data rather than the held-out test labels.

**Treat preprocessing as part of the model.** The 29 output columns depend on the fitted scaler, encoder categories, and ordering. Seven numeric columns plus 22 one-hot columns form the shared contract. Reordering or refitting them can break both the classical models and the downstream MLP.

**Verify when a feature becomes available.** `ClaimStatus` is excluded to avoid using a post-adjudication outcome. `Cluster` remains highly predictive but its construction is undocumented. A strong metric is not evidence that feature provenance is safe.

**Read plots according to what their code computes.** The monthly plot pools calendar months across years. The specialty and cluster figures show counts. Global feature importances describe fitted-model associations and do not explain the cause of a particular claim's outcome.

## Current limitations and next work

| Limitation | Evidence or consequence | Suggested next step |
|---|---|---|
| One supplied workbook | 4,500 rows from a single snapshot | Validate on independently obtained later claims |
| Random split | No temporal/group holdout in preprocessing | Evaluate chronological and appropriate group splits |
| Unknown `Cluster` provenance | No cluster-generation code in Approach 1; 267/270 frauds are in cluster 1 | Audit how it was built and evaluate an ablation without it |
| Small positive test sample | Only 40 fraud cases | Report uncertainty and evaluate more positive cases |
| No calibration audit | `predict_proba` scores have no independent calibration assessment | Assess reliability on validation data before treating scores as probabilities |
| No business-cost policy | Metrics use estimator class predictions | Compare thresholds using review cost and missed-fraud cost on validation data |
| No demographic fairness audit | Age, gender, income, and marital/employment fields are used | Evaluate subgroup errors and the suitability of these predictors |
| Limited explanations | Two global importance figures | Add validated local explanations and permutation-based checks |
| Incomplete environment record | Requirements use lower bounds; log lacks full package versions | Record a reproducible dependency lock and artifact hashes |
| Legacy script paths | All six helpers expect `project/` | Make paths relative to the script location in a future implementation change |
| Notebook generation defects | Literal placeholders, missing model imports, no executed outputs | Repair templates before using them as executable deliverables |
| Classical serving absent | No Approach 1 website directory or HTTP service | Implement and verify a dedicated prediction interface if needed |

## Shared application boundary

The broader repository contains frontend, Node, FastAPI, and agentic code. Their presence does not establish a working end-to-end classical prediction path. The specific dependency verified here is that `ml-service` loads Approach 1's fitted preprocessor and checks its ordered names against Approach 2's contract before MLP inference.

Changes to that preprocessing artifact require coordination with consumers. Keep Approach 1's saved comparison baseline intact when implementing another approach. This documentation update changes documentation only.

## Updating the documentation

After an intentional new training run:

1. Check the workbook hash, row counts, label counts, missing values, and split sizes.
2. Check that feature names, transformed widths, and downstream contracts agree.
3. Read `model_results.csv`, `best_model.txt`, `training_metrics.json`, and regenerated error counts together.
4. Refresh the tables in [Results](05_results.md), [KEY_RESULTS.md](../KEY_RESULTS.md), and [results.md](../results.md). Retain the distinction between validation selection and test reporting.
5. Update setup instructions only for scripts and services that exist and have been verified.
6. Check relative links and the historical `docs/` entry points, and record the documentation update date.

Markdown tables are maintained snapshots; the training scripts do not automatically update them. Historical architecture illustrations should be labeled as design context when shown in reports.
