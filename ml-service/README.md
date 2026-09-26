# Python ML Service

FastAPI inference service for Mini-Project 2.

## Current status

The following endpoints are implemented and verified:

```text
GET /health
POST /predict
```

The prediction path is:

```text
Raw claim
	↓
Validated ClaimInput
	↓
Read-only fitted transformer from Approch-1
	↓
29-feature frozen contract
	↓
Existing Approch-2-DL MLP checkpoint
	↓
PredictionResponse
```

The service does not retrain or modify the MLP. It reuses the existing `MLP` class and loads `Approch-2-DL/models/mlp/mlp_best.pt`. The fitted preprocessing transformer is read from `Approch-1/data/processed/preprocessor.joblib`; `Approch-1` remains unchanged.

### Health endpoint

```text
GET /health


Expected response:

```json
{"status":"ok"}
```

### Prediction endpoint

`POST /predict` accepts the model-relevant raw claim fields below. Optional identifiers, codes, location, and `ClaimStatus` are accepted only as descriptive metadata and are excluded from model input. `ClaimLegitimacy` is never accepted because it is the training target.

Required model-input fields:

- `ClaimAmount`
- `ClaimDate`
- `PatientAge`
- `PatientGender`
- `ProviderSpecialty`
- `PatientIncome`
- `PatientMaritalStatus`
- `PatientEmploymentStatus`
- `ClaimType`
- `ClaimSubmissionMethod`
- `Cluster`

`ClaimDate` is transformed into `ClaimYear`, `ClaimMonth`, and `ClaimDayOfWeek`. The fitted transformer then produces the exact frozen 29-feature ordering.

Example request:

```json
{
	"ClaimAmount": 794.07,
	"ClaimDate": "2024-03-04",
	"PatientAge": 75,
	"PatientGender": "M",
	"ProviderSpecialty": "Orthopedics",
	"PatientIncome": 99260.16,
	"PatientMaritalStatus": "Widowed",
	"PatientEmploymentStatus": "Employed",
	"ClaimType": "Emergency",
	"ClaimSubmissionMethod": "Paper",
	"Cluster": 0
}
```

Example response shape:

```json
{
	"model_name": "MLP",
	"model_version": "mlp_best.pt",
	"prediction": "Legitimate",
	"fraud_score": 0.0,
	"score_is_calibrated": false,
	"feature_count": 29,
	"feature_contract": "Approch-1 frozen 29-feature leakage-excluded baseline"
}
```

`fraud_score` is the MLP sigmoid output and is not a calibrated probability. Invalid or missing required fields return HTTP 422. Missing or incompatible model artifacts fail clearly during service loading.

Additional DL models, explainability, agentic AI, Node → FastAPI integration, and React → Node integration are not implemented yet.

## Run

Use the existing virtual environment only:

```powershell
& '.\Approch-2-DL\.venv\Scripts\python.exe' -m pip install -r ml-service\requirements.txt
& '.\Approch-2-DL\.venv\Scripts\python.exe' -m uvicorn app.main:app --app-dir ml-service --host 127.0.0.1 --port 8000
```
