# Setup, reproduction, and local inference

[Documentation index](README.md)

Run commands from the repository root, the folder containing `Approch-1/`. This guide uses PowerShell. The declared requirements recommend Python 3.11 or newer; the saved training log records Python 3.11.2. Dependencies have lower bounds rather than exact pins, so a fresh install is not an exact reconstruction of the original environment.

## Create an environment

```powershell
python --version
python -m venv .venv-approach1
& '.\.venv-approach1\Scripts\python.exe' -m pip install -r '.\Approch-1\requirements.txt'
```

Use a compatible installed Python interpreter for the `venv` command. Calling the environment's executable directly avoids requiring PowerShell activation. The requirements also list Flask, PDF, and presentation libraries; their corresponding website/report generators are absent from this checkout.

## Inspect existing artifacts

These commands read saved outputs without retraining:

```powershell
Get-Content '.\Approch-1\evaluation\best_model.txt'
Import-Csv '.\Approch-1\evaluation\model_results.csv' |
    Format-Table Model, Val_F1, Test_F1, Test_ROC_AUC, Test_PR_AUC
(Get-Content '.\Approch-1\data\processed\feature_names.txt').Count
```

Expected selected model: `GradientBoosting`. Expected feature count: `29`.

## Legacy path compatibility

The six scripts in `Approch-1/scripts/` still open and write paths beginning with `project/`. Invoking them as `python Approch-1/scripts/preprocess.py` does not change those internal paths. A Windows directory junction can map the expected name to the actual folder without editing the scripts.

First confirm that `project` does not already exist, including as a broken link. If it exists, inspect its target before proceeding; the following command deliberately refuses to replace it:

```powershell
if (Get-Item -LiteralPath '.\project' -Force -ErrorAction SilentlyContinue) {
    throw 'The project path already exists. Inspect it before running the legacy scripts.'
}
$approach1Target = (Resolve-Path -LiteralPath '.\Approch-1').Path
New-Item -ItemType Junction -Path '.\project' -Target $approach1Target
```

After creating the junction once, keep using the repository root as the working directory. On a system with symbolic-link support, an equivalent root-level `project` link to `Approch-1` serves the same purpose. The local inference example below uses actual paths and does not need the junction.

## Regenerate artifacts

The following commands overwrite preprocessing, models, metrics, figures, logs, and notebook templates through the junction. Rebuilding the preprocessing artifacts also changes a dependency consumed by `ml-service`; preserve the existing baseline if it is being used for comparisons.

```powershell
& '.\.venv-approach1\Scripts\python.exe' '.\Approch-1\scripts\preprocess.py'
& '.\.venv-approach1\Scripts\python.exe' '.\Approch-1\scripts\eda_plots.py'
& '.\.venv-approach1\Scripts\python.exe' '.\Approch-1\scripts\train_all_models.py'
& '.\.venv-approach1\Scripts\python.exe' '.\Approch-1\scripts\comparison_plots.py'
& '.\.venv-approach1\Scripts\python.exe' '.\Approch-1\scripts\error_analysis.py'
& '.\.venv-approach1\Scripts\python.exe' '.\Approch-1\scripts\create_notebooks.py'
```

The notebook generator produces templates with known placeholders and missing imports. It is optional for training and does not yield executable, verified notebooks without manual repair. There are no `generate_pdf.py`, `generate_ppt.py`, or `progress_tracker.py` scripts to run here.

## Predict with an existing classical model

This example uses `best_model.txt`, the fitted transformer, and a saved estimator. It writes no model or dataset files. Run it from the repository root in an environment compatible with the saved joblib artifacts:

```powershell
@'
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

root = Path('Approch-1')
model_name = (root / 'evaluation/best_model.txt').read_text().strip()
preprocessor = joblib.load(root / 'data/processed/preprocessor.joblib')
model = joblib.load(root / 'models' / f'{model_name}.joblib')

claim = pd.DataFrame([{
    'ClaimAmount': 794.07,
    'PatientAge': 75,
    'PatientIncome': 99260.16,
    'PatientGender': 'M',
    'ProviderSpecialty': 'Orthopedics',
    'PatientMaritalStatus': 'Widowed',
    'PatientEmploymentStatus': 'Employed',
    'ClaimType': 'Emergency',
    'ClaimSubmissionMethod': 'Paper',
    'Cluster': 0,
    'ClaimDate': '2024-03-04',
}])
dates = pd.to_datetime(claim.pop('ClaimDate'), errors='raise')
claim['ClaimYear'] = dates.dt.year
claim['ClaimMonth'] = dates.dt.month
claim['ClaimDayOfWeek'] = dates.dt.dayofweek

features = preprocessor.transform(claim)
names = (root / 'data/processed/feature_names.txt').read_text().splitlines()
produced = [name.split('__', 1)[-1] for name in preprocessor.get_feature_names_out()]
if names != produced or features.shape != (1, 29):
    raise ValueError('Feature contract mismatch')
if not np.isfinite(features).all():
    raise ValueError('Non-finite features')

label = int(model.predict(features)[0])
fraud_column = list(model.classes_).index(1)
score = float(model.predict_proba(features)[0, fraud_column])
print({
    'model_name': model_name,
    'prediction': 'Fraud' if label == 1 else 'Legitimate',
    'fraud_score': score,
    'score_is_calibrated': False,
    'feature_count': features.shape[1],
})
'@ | & '.\.venv-approach1\Scripts\python.exe' -
```

To compare another classical model, set `model_name` to `LogisticRegression`, `RandomForest`, `XGBoost`, `SVM`, or `GradientBoosting`. Preserve `model.predict` for class labels, particularly for SVM. The example's numeric values are illustrative, not a claim that every future dataset uses the same units or ranges.

## Website and API status

Approach 1 has no `website/app.py` in this checkout, so there is no Approach 1 Flask startup command or supported port-5000 API. The shared FastAPI `/predict` endpoint loads the Approach 2 MLP after reusing this approach's transformer. See the separate [ML service instructions](../../ml-service/README.md) for its dependencies and current entry point.

## Troubleshooting

| Symptom | Explanation and action |
|---|---|
| Missing `project/data/...` | Legacy relative paths are unresolved; check the junction target and working directory |
| Missing package | Install this approach's requirements using the same interpreter that runs the script |
| joblib load error or version warning | Saved estimators depend on their library versions; inspect the training log and use a compatible environment |
| Unexpected feature count/order | Use the saved transformer and ordered feature names together; do not create a new encoder for inference |
| Missing required column | Supply all 13 fields after date expansion as a named DataFrame |
| Unknown category | The saved encoder ignores it; validate inputs when building an application |
| Notebook placeholders such as `{class_def}` | Known generator limitation; use `train_all_models.py` for reproduction |
| Missing PDF, PPT, or website | Those deliverables and generators are absent; use the maintained Markdown documentation |
