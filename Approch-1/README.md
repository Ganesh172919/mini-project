# Approach 1 — Traditional Machine Learning for Medical Insurance Claim Fraud Detection

**Institution:** IIIT Dharwad · B.Tech (Data Science & AI)
**Faculty Adviser:** Prof. Ramesh Athe
**Team:** B Varshith (23BDS011) · M Jagadeshwar (23BDS033) · J Ganesh (23BDS024)

> One sentence: five classical classifiers are trained on the single Excel dataset
> that ships with this repository, scored once on a frozen 675-claim test block, and
> served through a localhost Flask website that lets a claims officer run any of the
> five models on a hand-entered claim.

---

## 1. Problem statement

Indian health insurers pay out crores of rupees every year on claims that should
never have been settled — inflated bills, phantom procedures, up-coded treatments
and organised rings around a single provider. Manual scrutiny does not scale: a
typical insurer receives thousands of claims a day and can adjudicate a few
hundred. This project builds the *first* line of automation: given the fields
known **at claim intake**, assign every claim a fraud probability so that human
investigators spend their time where it actually pays.

Framed as supervised binary classification:

| | |
|---|---|
| input | one insurance claim (amount, patient demographics, provider, claim type, date, cluster) |
| output | `P(fraud) ∈ [0, 1]` and, at the operating threshold, `Legitimate` / `Fraud` |
| target | `ClaimLegitimacy` (Fraud = 1, Legitimate = 0) |

## 2. Dataset (the only data used)

| property | value |
|---|---|
| file | `Health Insurance Fraud Claims.xlsx` (repository root; a copy lives in `data/raw/`) |
| rows × columns | 4,500 × 19 |
| fraud / legitimate | 270 / 4,230 (6.0 % positive class) |
| duplicates, missing cells | none that survive the audit in `docs/02_dataset_analysis.md` |
| external data | **none** — no Kaggle, no synthetic rows, no generated records |

Engineered from the raw columns: `ClaimYear`, `ClaimMonth`, `ClaimDayOfWeek` (from
`ClaimDate`), plus the repository's own `Cluster` segmentation. Dropped as
near-unique identifiers that would only memorise: `ClaimID`, `PatientID`,
`ProviderID`, `DiagnosisCode`, `ProcedureCode`, `ProviderLocation`, `ClaimDate`.

**Leakage control (final verification pass).** `ClaimStatus`
(Approved / Denied / Pending) is the insurer's *post-adjudication* outcome — it is
written after the investigation this model is meant to support. It was removed from
the feature registry in `scripts/preprocess.py`, from the web form, and from every
narrative artifact. The frozen registry is **29 columns** (7 scaled numeric +
22 one-hot), which is exactly the registry Approach 2 is scored on, so the two
approaches are now compared apples-to-apples.

## 3. Pipeline

```
Health Insurance Fraud Claims.xlsx
        │
        ▼  preprocess.py  ─── date expansion, drop IDs + leaky ClaimStatus
        │                     stratified 70/15/15 (random_state=42)
        │                     StandardScaler + OneHotEncoder fitted on TRAIN only
        ▼
   data/processed/*.npy, preprocessor.joblib, feature_names.txt
        │
        ▼  train_all_models.py ── 5 estimators, class-weighted for the 6 % prior
        │
        ├── evaluation/model_results.csv|json   (validation + test metrics)
        ├── evaluation/cm_*.png, roc_*.png, pr_*.png
        ├── models/*.joblib
        │
        ▼  comparison_plots.py / error_analysis.py
        ├── feature_importance_*.png, model_comparison_*.png, error_analysis_fp_fn.png
        ├── evaluation/result_analysis.md|txt   (FP / FN dossier per model)
        │
        ▼  generate_pdf.py / generate_ppt.py / progress_tracker.py / create_notebooks.py
        ├── outputs/project_documentation.pdf   (23 pages)
        ├── outputs/presentation.pptx           (19 slides)
        ├── outputs/progress_tracker.pdf
        └── notebooks/<Model>.ipynb             (one per model)
        │
        ▼  website/app.py (Flask, localhost:5000)
           form → validation → preprocessor → selected model → verdict + probability
```

Split sizes and fraud counts — train 3,150 (189 fraud) · validation 675 (41) ·
test 675 (40). Preprocessing is fitted on the training block only; the test block
is scored exactly once per model, and the winning model is chosen on **validation**
F1, never on test.

## 4. The five models and why they were chosen

| model | hyperparameters (as executed) | why this model here |
|---|---|---|
| LogisticRegression | `max_iter=2000`, `class_weight='balanced'`, `lbfgs`, `seed=42` | transparent, monotone baseline; a regulator can read the coefficients |
| RandomForest | `n_estimators=300`, `class_weight='balanced'`, `seed=42` | tabular workhorse, robust to outliers, free feature importances |
| XGBoost | `eval_metric='logloss'`, `scale_pos_weight=2961/189`, `seed=42` | gradient boosting with explicit imbalance correction — usually the ceiling on tabular data |
| SVM (RBF) | `probability=True`, `class_weight='balanced'`, `seed=42` | non-linear margin model; the honest "does boosting really add anything?" control |
| GradientBoosting (`HistGradientBoostingClassifier`) | `class_weight='balanced'`, `seed=42` | fast histogram boosting, bins the wide amount/income ranges without manual binning |

All five are class-weighted (or scale-pos-weighted) because only 6 % of claims are
fraudulent; with an unweighted objective the models would simply predict
"legitimate" and score 94 % accuracy while catching nothing.

## 5. Results — frozen test block, 675 claims (40 fraud)

Executed by `python project/scripts/train_all_models.py`; every number below is
written straight into `evaluation/model_results.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.9644 | 0.6250 | 1.0000 | 0.7692 | 0.9950 | 0.9186 | 24 | 0 |
| RandomForest | 0.9985 | 1.0000 | 0.9750 | 0.9873 | 0.9999 | 0.9994 | 0 | 1 |
| **XGBoost** | 0.9985 | 0.9756 | 1.0000 | **0.9877** | **1.0000** | **1.0000** | 1 | 0 |
| SVM (RBF) | 0.9778 | 0.7451 | 0.9500 | 0.8352 | 0.9955 | 0.9252 | 13 | 2 |
| GradientBoosting | 0.9985 | 1.0000 | 0.9750 | 0.9873 | 0.9847 | 0.9764 | 0 | 1 |

`best_model.txt` holds **GradientBoosting**, selected on validation F1 (1.0000).
XGBoost is the strongest on the test block (PR AUC 1.0000, 1 false alarm, 0 missed
fraud). The gap between the two is inside the noise of a 40-positive test block, so
the website exposes all five rather than hard-coding a single winner.

Reading the numbers like an insurer rather than like a leaderboard:

* **LogisticRegression** misses no fraud at all — the right model if a missed fraud
  costs far more than an investigation — but it sends 24 clean claims to the
  investigation desk (precision 0.625).
* **XGBoost / RandomForest / GradientBoosting** reduce the desk's workload to
  0–1 false alarms while still catching 39–40 of the 40 frauds.
* **SVM** sits in the middle: 13 false alarms and 2 missed frauds.
* Accuracy is nearly useless as a discriminator here (0.964 → 0.999 across models);
  precision, recall, PR AUC and the FP/FN counts are what a claims desk pays for.

## 6. Explainability

* `evaluation/feature_importance_RandomForest.png` and
  `evaluation/feature_importance_XGBoost.png` — impurity-based global importances
  from the two tree ensembles.
* `evaluation/model_comparison_f1.png`, `evaluation/model_comparison_roc.png` —
  the five models side by side on the metrics that matter for fraud.
* `evaluation/error_analysis_fp_fn.png` + `result_analysis.md` — the FP/FN dossier
  that drives the deployment recommendation.
* `notebooks/<Model>.ipynb` — one notebook per model with the real run outputs.
* The website returns the model name, the verdict and the fraud probability for
  every request, and appends each call to `website/prediction_log.txt` for audit.

**Dominant drivers (measured):** `ClaimAmount`, then the engineered `Cluster`
column, then the temporal features. `Cluster` carries most of the signal in this
dataset, which is why the recommendation in `result_analysis.md` is to validate on
a future time period before trusting these numbers in production.

## 7. Running everything (localhost only — no Docker, no CI)

```bash
# from the repository root
python3 -m venv .venv && source .venv/bin/activate
pip install -r project/requirements.txt

python project/scripts/preprocess.py         # split + fitted preprocessor
python project/scripts/eda_plots.py          # 10 EDA PNGs
python project/scripts/train_all_models.py   # train 5 models + metrics + curves
python project/scripts/comparison_plots.py   # comparison + importance PNGs
python project/scripts/error_analysis.py     # FP/FN dossier
python project/scripts/create_notebooks.py   # per-model notebooks
python project/scripts/generate_pdf.py       # outputs/project_documentation.pdf
python project/scripts/generate_ppt.py       # outputs/presentation.pptx
python project/scripts/progress_tracker.py   # outputs/progress_tracker.pdf

python project/website/app.py                # → http://localhost:5000
```

Website routes: `/` (predict with a selected model), `/results` (metrics table),
`/eda` (exploratory plots), `/compare` (all five models on the same input),
`/docs/<file>` and `/outputs/<file>` (serve the generated documents).

Quick API check — all five models on one claim:

```bash
for m in LogisticRegression RandomForest XGBoost SVM GradientBoosting; do
  curl -s -X POST http://localhost:5000/predict \
    --data-urlencode "model_name=$m" \
    -d "ClaimAmount=9000&PatientAge=64&PatientIncome=30000&ClaimYear=2024&ClaimMonth=7&ClaimDayOfWeek=2&PatientGender=M&ProviderSpecialty=Neurology&PatientMaritalStatus=Widowed&PatientEmploymentStatus=Retired&ClaimType=Inpatient&ClaimSubmissionMethod=Paper&Cluster=1"
done
```

## 8. Limitations

1. **Single source, single snapshot.** One 4,500-row Excel file with no temporal
   holdout; a future-time-period validation is the single most valuable next step.
2. **`Cluster` is doing a lot of the work.** It is an engineered column that
   correlates very strongly with fraud; if it is produced by a rule that later
   changes, the models must be retrained.
3. **40 positives in the test block.** Confidence intervals on every test metric
   are wide — differences of one or two claims are not real differences.
4. **No calibration layer here.** These models output ranking-quality scores;
   Approach 2 adds isotonic/temperature calibration and an explicit ECE audit, and
   that is where probability-as-money should be read from.
5. **No provider history features.** Repeat-billing velocity and per-provider
   denial rates are the classic fraud signals that this dataset does not contain
   (Approach 3's indicator layer is the place to add them).

## 9. Architecture for Approaches 2 and 3

`website/templates/compare.html` already carries clearly-labelled placeholder cards
for the deep-learning zoo (Approach 2) and the indicator/rules verifier
(Approach 3). Both are implemented in this repository as standalone FastAPI apps:

* `../approach_2_deep_learning_xai/` — 5 deep architectures, calibration, XAI,
  fairness, frozen leaderboard, FastAPI site on `:8010` (it imports this folder's
  `evaluation/model_results.csv` read-only, sha256-audited, as its classical anchor).
* `../approach_3_agent_ai/` — six-agent claim verification chain with a full XAI
  audit trail, FastAPI site on `:8020`.

The intended upgrade is a single front end that fans one claim out to all three
approaches and shows a unified verdict; each approach already exposes a JSON
prediction endpoint, so the integration is a routing layer, not a rewrite.

## 10. Deliverables checklist

| deliverable | path |
|---|---|
| fitted models (5) | `models/*.joblib` |
| metrics, per model | `evaluation/model_results.csv`, `.json` |
| confusion matrices, ROC/PR curves | `evaluation/cm_*.png`, `roc_*.png`, `pr_*.png` |
| explainability | `evaluation/feature_importance_*.png`, `notebooks/` |
| error analysis | `evaluation/result_analysis.md`, `error_analysis_fp_fn.png` |
| run logs | `logs/pipeline_run_<timestamp>.log`, `logs/training_metrics.json` (fit time, model size, per-model metrics) |
| documentation | `docs/01..06`, `docs/full_documentation.html` |
| report PDF | `outputs/project_documentation.pdf` (23 pages) |
| progress tracker PDF | `outputs/progress_tracker.pdf` |
| presentation | `outputs/presentation.pptx` (19 slides) |
| website | `website/app.py` (Flask, all 5 models live) |
| dependency file | `requirements.txt` |

---

**Data integrity statement.** Every table, plot, PDF and slide in this folder is
generated from `Health Insurance Fraud Claims.xlsx` by the scripts above. No
synthetic rows, no external datasets, no hand-typed results.
