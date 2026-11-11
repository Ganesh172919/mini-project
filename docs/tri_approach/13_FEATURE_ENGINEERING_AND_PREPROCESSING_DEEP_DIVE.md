# Volume 13 — Feature Engineering, Preprocessing Determinism & Vector Alignment Guide

> **Document Classification:** Data Transformation Specification & Cross-Language Encoding Parity  
> **Scope:** Exact column filtering, date-part decomposition, `StandardScaler` + `OneHotEncoder` pipeline invariants, policy-dependent feature widths (`29f` vs. `28f` `intake` vs. `no_amount`), and worked end-to-end encodings including `Row 1124`.  
> **Primary Ground-Truth Sources:** [`Approch-1/scripts/preprocess.py`](../../Approch-1/scripts/preprocess.py), [`Approch-1/data/processed/feature_names.txt`](../../Approch-1/data/processed/feature_names.txt), [`Approch-2-DL/processed/contract_metadata.json`](../../Approch-2-DL/processed/contract_metadata.json), [`submission/scripts/common.py`](../../submission/scripts/common.py), [`submission/scripts/export_browser.py`](../../submission/scripts/export_browser.py), [`submission/models/intake/contract.json`](../../submission/models/intake/contract.json), [`submission/results/dataset_audit.json`](../../submission/results/dataset_audit.json), [`submission/results/browser_parity.json`](../../submission/results/browser_parity.json), and [`ml-service/app/services/classical_service.py`](../../ml-service/app/services/classical_service.py).

---

## 1. End-to-End Preprocessing State Machine

Every vector fed into Track A (`Approch-1/`, `Approch-2-DL/`, `ml-service/`) and Track B (`submission/`) passes through an identical **four-stage deterministic ETL DAG** operating cleanly from the raw `19`-column `Health Insurance Fraud Claims.xlsx` workbook (`SHA-256: 952b6e...`):

```text
  [Raw Claim Input Record: 19 Workbook Columns / 11 Serving JSON Fields]
                                    |
                                    v
  Stage 1: Identifier & Outcome Leakage Purge
    DROP Identifiers/Sparse (6): ClaimID(4500), PatientID(4500), ProviderID(4500),
                                 DiagnosisCode(4495), ProcedureCode(4495), ProviderLocation(3876)
    DROP Leaky Outcome (1):      ClaimStatus (LEAKY_COLUMNS)
    OPTIONAL Policy Drop:        Cluster (dropped in `intake`), ClaimAmount (dropped in `no_amount`)
                                    |
                                    v
  Stage 2: Calendar Date Decomposition (`ClaimDate` -> 3 Numeric Integers)
    ClaimYear      = dt.year         (2022, 2023, or 2024 across 731 workbook dates)
    ClaimMonth     = dt.month        (1..12)
    ClaimDayOfWeek = dt.dayofweek    (Monday=0 .. Sunday=6 -- CRITICAL Python/JS Mapping!)
    DROP Raw String: ClaimDate
                                    |
                                    v
  Stage 3: Train-Only Parameter Fit (`N_train = 3,150` Random / `3,152` Temporal)
    Numeric Branch:     StandardScaler() -> z_j = (x_j - mu_j) / sigma_j   [fit on X_train ONLY]
    Categorical Branch: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                                    |
                                    v
  Stage 4: Concatenated Aligned Vector Emission
    Canonical Track A / baseline:  [num(0..6) | ohe(7..28)] -> float64/float32 shape (N, 29)
    Deployment `intake` Policy:    excludes Cluster -> length 28
    Browser JS Export (`:8080`):   mirrors mu, sigma & category lookup tables (0.0 bit drift over 1,350 checks)
```

---

## 2. Structural Rationale for Every Column Disposition (`19` Columns)

Inspecting `submission/results/dataset_audit.json` explains rigorously why specific columns from the `4,500`-row Excel corpus belong inside, or must stay strictly outside, statistical modeling matrices:

### 2.1 High-Cardinality Keys (`6` Excluded Columns)

| Column Name | Distinct Values / `4,500` Rows | Uniqueness Ratio | Why Included Models Would Collapse or Memorize |
| :--- | :---: | :---: | :--- |
| `ClaimID` | `4,500` | `100.00%` | Unique surrogate row key; zero shared recurrence across splits |
| `PatientID` | `4,500` | `100.00%` | Zero longitudinal repeat patients exist in corpus (`1` claim/patient) |
| `ProviderID` | `4,500` | `100.00%` | Zero recurring provider profiles across rows (`1` claim/provider) |
| `DiagnosisCode` | `4,495` | `99.89%` | Near-unique strings create `4,495` orthogonal sparse columns mostly unseen at test time |
| `ProcedureCode` | `4,495` | `99.89%` | Near-unique procedural IDs create empty test columns under OHE |
| `ProviderLocation` | `3,876` | `86.13%` | Extreme geographical granularity adds sparse overfitting without repeating hierarchy |

### 2.2 Leakage & Confounder Columns (`ClaimStatus` vs. `Cluster` vs. `ClaimAmount`)

1. **`ClaimStatus` (`LEAKY_COLUMNS`, Always Dropped):** Encodes `3` post-adjudication claim processing states unknown when a new claim arrives for screening.
2. **`Cluster` (`4` Levels `{0, 1, 2, 3}`, Conditional on Track/Scenario):**
   * Retains `idx 6` in canonical `29`-feature Track A (`Approch-1`, `Approch-2-DL`, `ml-service`) and Track B `baseline`.
   * **Dropped explicitly in Track B `intake` (`ADR-07`, `models/intake/contract.json`)** because `267` out of `270` frauds (`98.89%`) reside inside `Cluster == 1` (`1,100` claims -> `24.27%` fraud rate vs `0.00%` in Clusters `0` and `2`, `0.26%` in Cluster `3`). Real-time intake systems cannot request a dataset-wide unsupervised clustering output from an external submitting provider!
3. **`ClaimAmount` (`4,490` Unique Values, Dropped Only in `no_amount` Ablation):**
   * Fraudulent range: `\$7,465.07 – \$9,978.43` (`mean = \$8,733.5651`).
   * Legitimate range: `\$100.12 – \$9,997.20` (`mean = \$4,776.7978`).
   * Retained in production `intake` and canonical pipelines; ablated solely in Track B `no_amount` to prove that non-monetary demographics alone collapse classifier AP to `0.1869–0.2404`.

---

## 3. Categorical Taxonomy & Exact Alphabetical OHE Index Registry (`[7..28]`)

Because `OneHotEncoder` sorts observed string categories lexicographically during `.fit(X_train)`, the `22` binary indicators (`indices 7–28` in `Approch-1/data/processed/feature_names.txt`) occupy **immutable positional offsets**. Even a single column rename or swap scrambles weights inside linear hyperplanes, PyTorch `MLP`, `CNN1D` local `kernel_size=3` filters, and tree split feature indices.

| Categorical Feature | Unique Count | Exact Lexicographical Category Set (Indexed Order) | Track A Indices | Track B `intake` (`ex-Cluster`) Indices |
| :--- | :---: | :--- | :---: | :---: |
| **`PatientGender`** | `2` | `F`, `M` | `7..8` | `6..7` |
| **`ProviderSpecialty`** | `5` | `Cardiology`, `General Practice`, `Neurology`, `Orthopedics`, `Pediatrics` | `9..13` | `8..12` |
| **`PatientMaritalStatus`** | `4` | `Divorced`, `Married`, `Single`, `Widowed` | `14..17` | `13..16` |
| **`PatientEmploymentStatus`** | `4` | `Employed`, `Retired`, `Student`, `Unemployed` | `18..21` | `17..20` |
| **`ClaimType`** | `4` | `Emergency`, `Inpatient`, `Outpatient`, `Routine` | `22..25` | `21..24` |
| **`ClaimSubmissionMethod`** | `3` | `Online`, `Paper`, `Phone` | `26..28` | `25..27` |
| **Total Binary Columns** | **`22`** | *(Each row has strictly `6` ones and `16` zeros if all categories known)* | **`22` cols** | **`22` cols** |

### 3.1 Unknown Category Runtime Handling (`handle_unknown='ignore'`)

Both `Approch-1/scripts/preprocess.py` and Track B initialize `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. If a live caller submits an unseen string such as `"ProviderSpecialty": "Dermatology"`, scikit-learn emits `[0, 0, 0, 0, 0]` across indices `[9..13]` rather than throwing an unhandled `500 Internal Server Error` during transformation. Contrast this cleanly with **property key typos** (e.g., `"specialty": "Cardiology"`), which are caught upfront and rejected with HTTP `422` via Pydantic `extra="forbid"`.

---

## 4. Calendar Date Engineering & Cross-Language Pitfalls

Raw workbook timestamps (`ClaimDate`) cover **`731` unique calendar days** from `2022-07-09` through `2024-07-08`. Preprocessing extracts three scalar integers:

1. `ClaimYear` (`2022`, `2023`, `2024`)
2. `ClaimMonth` (`1` = January .. `12` = December)
3. `ClaimDayOfWeek` (**`pandas.Series.dt.dayofweek` convention**)

### 4.1 Critical Cross-Runtime Parity Trap: Python Monday=`0` vs. JavaScript Sunday=`0`!

When porting Python `ColumnTransformer` preprocessing into pure JavaScript inside `submission/scripts/export_browser.py` (`browser_parity.json`), engineers encounter a famous multi-language trap:

| Weekday | `pandas.dt.dayofweek` (Expected by Models) | Native JS `Date.getUTCDay()` / `getDay()` | Correct Modulo Conversion Needed in JS |
| :--- | :---: | :---: | :--- |
| **Sunday** | `6` | `0` | `(jsDay + 6) % 7` → `(0 + 6) % 7 = 6` |
| **Monday** | **`0`** | `1` | `(1 + 6) % 7 = 0` |
| **Tuesday** | `1` | `2` | `(2 + 6) % 7 = 1` |
| **Wednesday** | `2` | `3` | `(3 + 6) % 7 = 2` |
| **Thursday** | `3` | `4` | `(4 + 6) % 7 = 3` |
| **Friday** | `4` | `5` | `(5 + 6) % 7 = 4` |
| **Saturday** | `5` | `6` | `(6 + 6) % 7 = 5` |

Moreover, parsing `"YYYY-MM-DD"` strings via `new Date("2024-03-19")` in client JavaScript uses UTC midnight (`getUTCDay()`) rather than local browser timezone (`getDay()`), avoiding negative-UTC timezone off-by-one shifts! In `browser_parity.json`, testing all **`1,350` validation + test claims** proved **`max_err = 0.0` exact equivalence** between Track B JavaScript preprocessing and Python scikit-learn outputs.

---

## 5. `CNN1D` Local Adjacency Sensitivity & `Temporal` Drift Coupling

Notice two unexpected downstream consequences of feature construction revealed across Volumes `04` and `07`:

1. **Convolutional Neighborhood Effect (`CNN1D`):** Inside [`Approch-2-DL/scripts/submission_models.py`](../../Approch-2-DL/scripts/submission_models.py), `Conv1d(1, 8, kernel_size=3, padding=1)` sweeps `3`-element sliding windows across aligned indices `0..28`. Because indices `[0, 1, 2]` place `ClaimAmount`, `PatientAge`, and `PatientIncome` contiguously — directly adjacent to `[3, 4, 5]` (`ClaimYear, Month, DayOfWeek`) and `Cluster` (`[6]`) — local filters automatically couple monetary scale, age, income, and cluster!
2. **`ClaimYear` Extrapolation Effect (`temporal` Split):** Training runs in Track B `temporal` see dates up through `2023-11-20` (centered mostly around `2022–2023`). When standardized test claims arrive from `2024-03-11` through `2024-07-08`, feature `idx 3` (`ClaimYear = 2024`) assumes persistently high positive $z$-scores unseen in calibration bounds, pulling continuous linear weights and neural representations off-manifold (`MLP` `0.8581`, `ResMLP` `0.7979`, `SVM` `0.8396`) while decision trees cleanly threshold around `ClaimAmount` and remain at `1.0000` AP/F1!

---

## 6. Worked Boundary Transformations (`Row 1124` Fraud vs. Minimum Legit Claim)

Applying the canonical rules side-by-side clarifies exactly what enters Track A (`29f`) vs. Track B `intake` (`28f`) classifiers on extreme workbook examples audited in `dataset_audit.json` and `analysis.json`:

| Pipeline Attribute | Fraud Boundary Case (`Row 1124`, `analysis.json`) | Global Legitimate Minimum (`dataset_audit.json`) |
| :--- | :--- | :--- |
| **`ClaimAmount`** | **`\$7,465.07`** *(exact fraud lower boundary)* | **`\$100.12`** *(far outside fraud zone)* |
| **`PatientAge`** | `39` | Domain variable |
| **`ClaimType`** | `"Outpatient"` → sets `ClaimType_Outpatient = 1.0` | Categorical OHE bit `1.0` |
| **`Cluster` Handling** | Kept in `baseline` (`29f`); **dropped in `intake` (`28f`)** | Kept in `baseline`; **dropped in `intake`** |
| **`RandomForest` `intake` Score** | **`0.27`** (`< 0.4167` thr -> `FN`, yet `> all 635 legits` -> `AP=1.0`) | `< 0.27` (`TN` across all `635` benign test rows) |
| **`AgenticEnsemble-v1` Stub** | Evaluates `0.01*7465.07 + 0.005*39 - 2.0` -> **`1.00`** | At `\$100.12` + age `40`: `1.0012 + 0.2 - 2.0` -> **`0.00`** *(but clamps any legit >`\$300` to `1.00`!)* |
| **`ClassicalMLPage` Offline Fallback** | `(7465.07*0.00003) + 0.039 - 1.5 = -1.237` -> **`0.01`** | Evaluates -> **`0.01`** |
