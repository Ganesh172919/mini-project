# Medical Insurance Claim Fraud Detection submission

This package covers **Approach 1 and Approach 2 only**. The final artifacts are in `deliverables/`:

- `Presentation.pptx`: 18 editable slides, including native tables and charts.
- `Research_Paper.docx`: 7-page research manuscript.
- `Project_Report.docx`: 15-page technical report with formulas and demonstration instructions.

## Demonstration

Private hosted website: https://claim-fraud-lab-ravi-2026.clearbell16.chatgpt.site

To run the local website and independent Python API from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File submission/START_DEMO.ps1
```

Open http://127.0.0.1:8080. Use the **Intake** contract, load either workbook example, and test the selected model. **Compare all 8** evaluates every algorithm. The Approach buttons select the classical or neural family. Results contains four measured experiments. Prediction JSON downloads locally.

The static site computes scores using the actual exported fitted parameters. Claim inputs stay in the browser. The local API is an independent Python inference path, available at `/api/predict`; the website does not require a running Python API for browser inference. `/docs` exposes the local API schema.

The hosted copy is owner-private. For an evaluator without that account, use the local demonstration or serve `website/dist/` with any static HTTP server. No model-service credentials are required.

## Scope and results

Approach 1: Logistic Regression, Random Forest, XGBoost, SVM and histogram Gradient Boosting.

Approach 2: MLP, Residual MLP and 1D CNN. The latter two are new implementations under `Approch-2-DL/scripts/submission_models.py`.

All eight algorithms were trained in four experiments (32 fitted combinations):

| Experiment | Encoded features | Partition |
|---|---:|---|
| Original baseline | 29 | Stratified random, seed 42 |
| Intake without Cluster | 28 | Same random rows |
| Without Cluster and ClaimAmount | 27 | Same random rows |
| Chronological intake | 28 | Complete dates kept together |

The default intake Random Forest has test AP **1.0000**, precision **1.0000**, recall **0.9750**, F1 **0.987342**, and confusion counts TN=635, FP=0, FN=1, TP=39 at threshold **0.4166667**. Selection uses validation AP, validation F1 and the declared model order, never test performance. The amount ablation reduces Random Forest AP to **0.206596**, exposing substantial dependence on amount-related patterns.

These are retrospective internal benchmarks. The random test split has appeared in earlier results. Data origin, currency, label creation and Cluster construction are undocumented. All scores are uncalibrated. The project supports a submission demonstration, with external predictive validity still unestablished.

## Rebuild models and verify

The existing `.submission-venv` is ready on this computer. For a fresh Windows installation, install Python 3.12 and run:

```powershell
python -m venv .submission-venv
.submission-venv/Scripts/python.exe -m pip install -r submission/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Then run from the repository root:

```powershell
.submission-venv/Scripts/python.exe submission/scripts/run_pipeline.py
.submission-venv/Scripts/python.exe submission/scripts/export_browser.py
.submission-venv/Scripts/python.exe submission/scripts/make_figures.py
.submission-venv/Scripts/python.exe submission/scripts/evaluate_legacy.py
.submission-venv/Scripts/python.exe submission/scripts/verify.py
node submission/scripts/check_browser.mjs
```

Training resolves all paths from the script location. It does not depend on the historical `project/` directory in old scripts. Retraining replaces only fresh artifacts under `submission/models`, `submission/results` and `submission/data`. It does not overwrite the original workbook or historical model files. Run browser export after retraining and refresh any open site to load the new model parameters.

Validation evidence:

- 128 metric rows independently recomputed from stored predictions, including count formulas, grouped-score AP and rank-based ROC AUC.
- 1,350 raw-claim preprocessing comparisons: exact float32 agreement.
- 10,800 browser/Python model-score comparisons: maximum absolute difference below 0.000001, zero validation-threshold decision mismatches.
- 32 valid API prediction requests, four invalid-input checks and five static routes verified.
- Six historical model artifacts evaluated read-only. The historical XGBoost pickle emits an older-serialization warning, recorded in the audit.

## Artifact map

| Location | Contents |
|---|---|
| `models/<scenario>/` | Preprocessor, feature contract, five joblib models and three neural checkpoints |
| `results/metrics.csv` | Validation/test metrics for default and validation-selected thresholds |
| `results/predictions.csv` | Every evaluation score and workbook row number |
| `results/dataset_audit.json` | Workbook fingerprint, distributions, split counts and date cutoffs |
| `results/registry.json` | Model family, feature count and frozen threshold |
| `results/selection.json` | Validation-only default model selection |
| `results/bootstrap.json` | Conditional test bootstrap intervals |
| `results/analysis.json` | Feature importance and selected-model errors |
| `data/*split_manifest.csv` | Workbook row and claim ID partition membership |
| `figures/` | Measured data plots and website screenshot |
| `website/dist/` | Portable static site, results and browser model parameters |
| `scripts/serve.py` | FastAPI service, input contract and static hosting |
| `requirements-lock.txt` | Exact measured Python environment |

## Rebuilding submission documents

`scripts/build_documents.py` generates the Word files from measured results and figures. Use the Codex bundled artifact Python runtime with python-docx. `scripts/build_presentation.mjs` uses the bundled `@oai/artifact-tool`, with runtime paths supplied as described at the top of that file. Regenerate and visually verify documents and slides after changes to the results. The current documents have been rendered in native Word and visually reviewed; PowerPoint package checks include chart workbook references, editable tables, fonts and successful first-party reimport.

## API example

Send `POST /api/predict` with the following shape. The complete claim fields are available in `website/dist/examples.json`:

```json
{
  "scenario": "intake",
  "model": "RandomForest",
  "policy": "validation_f1",
  "claim": {
    "ClaimAmount": 7820.52,
    "ClaimDate": "2024-07-08",
    "PatientAge": 96,
    "PatientIncome": 57595.11,
    "PatientGender": "F",
    "ProviderSpecialty": "Orthopedics",
    "PatientMaritalStatus": "Single",
    "PatientEmploymentStatus": "Employed",
    "ClaimType": "Inpatient",
    "ClaimSubmissionMethod": "Paper"
  }
}
```

The intake API rejects a supplied Cluster value. Baseline requires Cluster. Extra fields, the target, invalid dates and out-of-range values are rejected.
