# Dataset and preprocessing

[Documentation index](README.md)

## Source audit

The preprocessing script reads the copy of `Health Insurance Fraud Claims.xlsx` in `data/raw/`. Its SHA-256 matches the workbook at the repository root:

```text
952b6e23c9845b9086994c08bc81323575968c318f845ea69e3b9609614b6a45
```

| Property | Verified value |
|---|---|
| Worksheet | `Sheet1` |
| Raw shape | 4,500 rows x 19 columns |
| Target counts | 4,230 Legitimate; 270 Fraud |
| Fraud prevalence | 6.00% |
| Missing cells | 0 |
| Duplicate full rows | 0 |
| Claim dates | 9 July 2022 through 8 July 2024 |
| Distinct claim dates | 731 |

The pipeline uses this workbook only. The repository does not establish how the original workbook was collected or whether it represents a production insurance population.

## Raw field dictionary

| Field | Observed values or cardinality | Treatment |
|---|---|---|
| `ClaimID` | 4,500 distinct | Drop identifier |
| `PatientID` | 4,500 distinct | Drop identifier |
| `ProviderID` | 4,500 distinct | Drop identifier |
| `ClaimAmount` | Numeric; 4,490 distinct | Scale |
| `ClaimDate` | 731 distinct dates | Derive date features, then drop original |
| `DiagnosisCode` | 4,495 distinct | Drop high-cardinality code |
| `ProcedureCode` | 4,495 distinct | Drop high-cardinality code |
| `PatientAge` | Integers 0 through 99 | Scale |
| `PatientGender` | F, M | One-hot encode: 2 columns |
| `ProviderSpecialty` | Cardiology, General Practice, Neurology, Orthopedics, Pediatrics | One-hot encode: 5 columns |
| `ClaimStatus` | Approved, Denied, Pending | Exclude as a potential post-adjudication leakage field |
| `PatientIncome` | Numeric; 4,500 distinct | Scale |
| `PatientMaritalStatus` | Divorced, Married, Single, Widowed | One-hot encode: 4 columns |
| `PatientEmploymentStatus` | Employed, Retired, Student, Unemployed | One-hot encode: 4 columns |
| `ProviderLocation` | 3,876 distinct | Drop high-cardinality location |
| `ClaimType` | Emergency, Inpatient, Outpatient, Routine | One-hot encode: 4 columns |
| `ClaimSubmissionMethod` | Online, Paper, Phone | One-hot encode: 3 columns |
| `Cluster` | 0, 1, 2, 3 | Treat as numeric and scale |
| `ClaimLegitimacy` | Fraud, Legitimate | Encode target as 1, 0 |

## Transformation sequence

1. Read the workbook with pandas.
2. Parse `ClaimDate` with `errors='coerce'`; derive `ClaimYear`, `ClaimMonth`, and `ClaimDayOfWeek` (Monday = 0, Sunday = 6).
3. Remove identifiers, high-cardinality codes/location, and the original date.
4. Map the target to 0/1 and exclude `ClaimStatus`.
5. Split the resulting 13 input columns into train, validation, and test blocks.
6. Fit `StandardScaler` and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on training data only. Transform validation/test with the fitted transformer.
7. Save raw input CSVs, encoded NPY arrays, target files, the fitted transformer, and the ordered feature names.

The script contains no missing-value imputation, duplicate removal, category validation, oversampling, or log transformation. Malformed dates can become missing values; the lack of missing values in the supplied workbook is not a guarantee for new inputs. Unknown categorical values produce an all-zero encoding for that categorical family.

## Split ledger

Two `train_test_split` calls use stratification and `random_state=42`: a 70/30 split followed by a 50/50 split of the remaining 30%.

| Block | Rows | Fraud | Legitimate | Encoded shape |
|---|---:|---:|---:|---|
| Train | 3,150 | 189 | 2,961 | `(3150, 29)` |
| Validation | 675 | 41 | 634 | `(675, 29)` |
| Test | 675 | 40 | 635 | `(675, 29)` |

These are random stratified splits. They are not chronological or grouped by provider/patient.

## Frozen feature contract

Numeric output order:

```text
ClaimAmount, PatientAge, PatientIncome, ClaimYear,
ClaimMonth, ClaimDayOfWeek, Cluster
```

The six categorical families add `2 + 5 + 4 + 4 + 4 + 3 = 22` columns, giving **7 + 22 = 29** total outputs. [feature_names.txt](../data/processed/feature_names.txt) is the complete ordered contract. The saved Approach 2 feature-name list matches it exactly.

`X_*.csv` files contain the 13 input columns before encoding; `X_*.npy` files contain the transformed 29-column matrices. `y_*` files contain numeric targets. Refitting the encoder or changing column order can invalidate models and the downstream MLP contract.

## Leakage and provenance limits

`ClaimStatus` is deliberately excluded because the intended decision happens before adjudication. It remains available for exploratory plotting. Excluding this field does not prove that every remaining feature is available at intake: the workbook supplies `Cluster`, but Approach 1 contains no code explaining how it was produced. Its provenance and availability need verification before operational use.
