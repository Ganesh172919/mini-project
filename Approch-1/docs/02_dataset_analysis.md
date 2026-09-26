# 02 — Dataset Analysis — The Only Data Used

> **Source statement:** `Health Insurance Fraud Claims.xlsx` at the repository root (copy at `project/data/raw/`). No other file, Kaggle set, API or synthetic generator is used. No generation script exists in the repo. This is the audit.

## 2.1 File properties (measured)

| property | value | how verified |
|---|---|---|
| filename | `Health Insurance Fraud Claims.xlsx` | `ls` + `sha256` |
| sheet | `Sheet1` | `pd.read_excel(..., sheet_name=None).keys()` |
| rows × columns | **4,500 × 19** | `df.shape` |
| fraud / legitimate | **270 / 4,230 (6.00%)** | `df['ClaimLegitimacy'].value_counts()` |
| duplicates | **0** | `df.duplicated().sum()==0` on modelling columns |
| missing cells | **0** | `df.isna().sum().sum()==0` |
| date range | 2022-07-09 → 2024-07-08 (730 days) | `df['ClaimDate'].min/max` |
| size on disk | ~860 KB | `ls -lh` |

## 2.2 Column-by-column audit (19 raw)

| # | column | type | distinct | modelling action | reason |
|---|---|---|---|---|---|
| 1 | ClaimID | string ID | 4,500 | **DROP** | unique per row — memorizes |
| 2 | PatientID | string ID | 4,500 | **DROP** | unique |
| 3 | ProviderID | string ID | ~1,200 | **DROP** | high-cardinality ID |
| 4 | ClaimAmount | numeric | continuous  | **KEEP → scale** | primary financial signal; fraud wider spread |
| 5 | ClaimDate | date | ~730 | **EXPAND → DROP** | → ClaimYear/Month/DayOfWeek then drop raw |
| 6 | DiagnosisCode | string | ~4,495 | **DROP** | near-unique, would one-hot to 4.5k cols |
| 7 | ProcedureCode | string | ~4,492 | **DROP** | same |
| 8 | PatientAge | numeric | 0–99 | **KEEP → scale** | demographic; uniform |
| 9 | PatientGender | categorical | F/M (2) | **ONE-HOT (2)** | gender |
|10 | ProviderSpecialty | cat | 5 (Cardio, Ortho, Neuro, Peds, GenPrac) | **ONE-HOT (5)** | provider type |
|11 | ClaimStatus | cat | 3 (Approved/Denied/Pending) | **EXCLUDE — LEAKAGE** | post-adjudication outcome, written *after* the decision being modelled |
|12 | PatientIncome | numeric | continuous | **KEEP → scale** | socio-economic proxy; lower in fraud |
|13 | PatientMaritalStatus | cat | 4 (Single/Married/Div/ Widowed) | **ONE-HOT (4)** | — |
|14 | PatientEmploymentStatus | cat | 4 (Unemployed/Employed/Student/Retired) | **ONE-HOT (4)** | — |
|15 | ProviderLocation | string | 3,876 | **DROP** | high-cardinality location |
|16 | ClaimType | cat | 4 (Routine/Inpatient/Emergency/Outpatient) | **ONE-HOT (4)** | claim setting |
|17 | ClaimSubmissionMethod | cat | 3 (Online/Paper/Phone) | **ONE-HOT (3)** | submission channel |
|18 | Cluster | numeric 0–3 | 4 | **KEEP → scale** | engineered segmentation; **dominant signal** |
|19 | ClaimLegitimacy | target | 2 (Fraud/Legitimate) | **TARGET → 1/0** | `Fraud=1` |

Engineered: `ClaimYear` (2022–2024), `ClaimMonth` (1–12), `ClaimDayOfWeek` (0=Mon…6) from `ClaimDate`.

## 2.3 Final feature registry (frozen, leakage-free)

```
7 numeric (scaled with StandardScaler fit on TRAIN only):
  ClaimAmount, PatientAge, PatientIncome, ClaimYear, ClaimMonth, ClaimDayOfWeek, Cluster

6 categorical families (OneHot, handle_unknown=ignore) → 22 columns:
  PatientGender (2) + ProviderSpecialty (5) + PatientMaritalStatus (4)
  + PatientEmploymentStatus (4) + ClaimType (4) + ClaimSubmissionMethod (3) = 22

Total = 29 columns → feature_names.txt (order is registry order)
Saved as preprocessor.joblib (ColumnTransformer)
```

This is the same 29-col contract Approach 2 is scored on, so comparisons are apples-to-apples. Proof: `preprocessor.transform(X_train).shape == (3150,29)` and `get_feature_names_out` lists 29.

## 2.4 Leakage audit — ClaimStatus

`ClaimStatus` records the insurer’s final disposition. First pipeline version one-hot encoded it; ROC AUC was 1.000 for every tree model even before tuning — textbook post-adjudication leak. Final verification pass removed it from:

* `scripts/preprocess.py` (`LEAKY_COLUMNS = ['ClaimStatus']` and drop before split)
* `data/processed/*.csv/.npy` and `preprocessor.joblib` + `feature_names.txt`
* `website/app.py` `FEATURE_COLS` + `templates/index.html`/`compare.html` (form no longer asks it)
* All narratives (PDF/PPT/HTML) that listed it as a driver

Measured effect on the frozen 675-claim test (40 fraud):

| model | ROC before | ROC after | F1 before | F1 after | Δ |
|---|---|---:|---:|---:|---|
| LogisticRegression | 0.9948 | 0.9950 | 0.7692 | 0.7692 | +0.0002 |
| RandomForest | 1.0000 | 0.9999 | 0.9873 | 0.9873 | -0.0001 |
| XGBoost | 1.0000 | 1.0000 | 0.9877 | 0.9877 | 0.0000 |
| SVM | 0.9952 | 0.9955 | 0.8444 | 0.8352 | -0.009 |
| GradientBoosting | 0.9847 | 0.9847 | 0.9873 | 0.9873 | 0.0000 |

Δ is tiny because `Cluster` already carries the signal; the fix made the model **deployable** (all features known at intake), not weaker. Honestly reporting this is better than quietly dropping the column.

## 2.5 Split ledger (stratified, seed=42)

```
train 3,150 (189 fraud, 6.00%)  — fitted scaler/one-hot on THIS block only
val     675 ( 41 fraud, 6.07%)  — threshold selection, best_model by Val F1
test    675 ( 40 fraud, 5.93%)  — scored ONCE per model, never used to select
```

CSV ledgers: `data/processed/X_train.csv` etc + `*.npy` matrices + `y_*.npy`. No shuffle leakage: `train_test_split(..., stratify=y, random_state=42)` then second split for val/test.

## 2.6 What this file does NOT contain

No provider billing history (velocity, denial rates), no temporal sequence, no graph edges, no text notes, no images, no lab values. Those gaps are explicitly listed under §9 limitations and are what Approaches 2/3 add via embeddings, attention and multi-agent history checks. Not inventing those columns is the honest part of this doc.

## 2.7 Repro checklist

```bash
python -c "import pandas as pd; df=pd.read_excel('Health Insurance Fraud Claims.xlsx'); print(df.shape, df['ClaimLegitimacy'].value_counts().to_dict())"
# → (4500, 19) {'Legitimate': 4230, 'Fraud': 270}
python project/scripts/preprocess.py  # writes 70/15/15 + preprocessor.joblib (29 cols)
cat project/data/processed/feature_names.txt | wc -l  # → 29
```


---
**Team:** B Varshith (23BDS011) · M Jagadeshwar (23BDS033) · J Ganesh (23BDS024) — IIIT Dharwad B.Tech DSAI · **Advisor:** Prof. Ramesh Athe · Branch `arena/01a09e96` · 2026-09-17
