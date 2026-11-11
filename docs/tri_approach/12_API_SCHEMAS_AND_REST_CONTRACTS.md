# Volume 12 — API Schemas, REST Route Contracts & Validation Reference

> **Document Classification:** Cross-Tier Network & Schema Interchange Specification  
> **Service Coverage:** Tier 1 FastAPI (`:8000`), Tier 2 Express TypeScript (`:3000`), Prototype Agentic (`:8020`), Track B Controlled Demo (`:8080`).  
> **Primary Ground-Truth Sources:** [`ml-service/app/schemas/claims.py`](../../ml-service/app/schemas/claims.py), [`ml-service/app/api/predictions.py`](../../ml-service/app/api/predictions.py), [`backend/node-api/src/server.ts`](../../backend/node-api/src/server.ts), [`routes/health.ts`](../../backend/node-api/src/routes/health.ts), [`routes/ml.ts`](../../backend/node-api/src/routes/ml.ts), [`routes/chat.ts`](../../backend/node-api/src/routes/chat.ts), [`agentic/predict.ts`](../../backend/node-api/src/agentic/predict.ts), [`shared/schemas.ts`](../../backend/node-api/src/shared/schemas.ts), [`Approach-3-Agentic-AI/app/api/predict.py`](../../Approach-3-Agentic-AI/app/api/predict.py), [`frontend/src/types.ts`](../../frontend/src/types.ts), [`submission/scripts/serve.py`](../../submission/scripts/serve.py), and [`submission/results/verification.json`](../../submission/results/verification.json).

---

## 1. Unified Multi-Service Endpoint Directory

Every HTTP endpoint exposed across the four runnable processes in the repository is indexed below with exact binding port, origin behavior, and schema strictness:

| Port & Process | HTTP Method & Path | Handler Source File | Request Schema | Downstream Call / Behavior |
| :---: | :--- | :--- | :--- | :--- |
| **`:8000` FastAPI** | `GET /` / `GET /health` | `ml-service/app/main.py` / router | None | Service liveness & loaded model inventory readiness |
| **`:8000` FastAPI** | `POST /predict/ml` | `ml-service/app/api/predictions.py` | `ClaimRequest` (`extra="forbid"`) | Calls `classical_service.py` (selected Approach 1 `.joblib` model or all `5`) |
| **`:8000` FastAPI** | `POST /predict/dl` | `ml-service/app/api/predictions.py` | `ClaimRequest` (`extra="forbid"`) | Calls `dl_service.py` (selected Approach 2 `.pt` network + `Autoencoder` sigmoid) |
| **`:8000` FastAPI** | `POST /predict` | `ml-service/app/api/predictions.py` | `ClaimRequest` (`extra="forbid"`) | Unified default prediction endpoint inside `ml-service` |
| **`:3000` Express** | `GET /api/health` | `backend/node-api/src/routes/health.ts` L5–7 | None | Returns `{ "status": "ok" }` (`healthRouter`) |
| **`:3000` Express** | `POST /api/ml/*` (`/predict`, `/predict/ml`, `/predict/dl`) | `backend/node-api/src/routes/ml.ts` (`111` lines) | Claim JSON payload | Proxies cleanly to `http://127.0.0.1:8000/*`; normalizes FastAPI `422` & upstream errors |
| **`:3000` Express** | `POST /api/agentic/predict` | `backend/node-api/src/agentic/predict.ts` (`39` lines) | Lower/Pascal claim amount + age payload | Executes in-process `AgenticEnsemble-v1` + 4-stage trace + static XAI (`feature_count=9`) |
| **`:3000` Express** | `POST /api/chat` | `backend/node-api/src/routes/chat.ts` (`47` lines) | `{ message, context }` JSON | Reads `../../Approach-3-Agentic-AI/.env`, invokes `gemini-3.8-flash` |
| **`:8020` FastAPI** | `GET /`, `GET /health` | `Approach-3-Agentic-AI/app/main.py` | None | Standalone prototype health status |
| **`:8020` FastAPI** | `POST /api/v1/predict`, `GET /api/v1/audit` | `Approach-3-Agentic-AI/app/api/predict.py` | Claim fields | Heuristic scorer + `XAIExplainer` + RAM `audit_log` (subject to duplicate router warning!) |
| **`:8080` Static/Lab** | `GET /` & `5` static asset routes + `32` API checks | `submission/scripts/serve.py` | Track B JSON schema | Serves pure-JS inference dashboard + audited `submission/` scenario predictions |

---

## 2. Canonical Claim Input Schema (`ml-service` `ClaimRequest` & `extra="forbid"`)

Tier 1 (`ml-service/app/schemas/claims.py`) acts as the authoritative schema gatekeeper via Pydantic v2 `ConfigDict(extra="forbid")`. Notice the deliberate **PascalCase naming convention** matching exact workbook headers in `Health Insurance Fraud Claims.xlsx`:

### 2.1 Strict Request Field Specification (`11` Inbound Attributes -> `29` Engineered Features)

| Inbound JSON Property Name | Data Type | Allowed / Audited Domain Profile | Target Transformed Indices (`feature_names.txt`) | Validation Rule |
| :--- | :--- | :--- | :--- | :--- |
| **`ClaimAmount`** | `float` | `\$100.12` – `\$9,997.20` (`fraud min \$7,465.07`) | `idx 0` (`StandardScaler`) | Numeric; required on Track A (`ex-amount` only in Track B test) |
| **`PatientAge`** | `int` | `100` distinct ages in workbook | `idx 1` (`StandardScaler`) | Integer age |
| **`PatientIncome`** | `float` | Continuous claimant income (`4,500` uniques) | `idx 2` (`StandardScaler`) | Numeric income |
| **`ClaimDate`** | `str` (`ISO date`) | `"YYYY-MM-DD"` spanning `2022-07-09` .. `2024-07-08` | Parses into `idx 3, 4, 5` (`ClaimYear`, `ClaimMonth`, `ClaimDayOfWeek`) | Parsed via `pd.to_datetime`; Monday = `0` |
| **`Cluster`** | `int` | `{0, 1, 2, 3}` (`Cluster 1` = `98.89%` fraud) | `idx 6` (`StandardScaler`) | Required in canonical `29f` Track A; **excluded** in Track B `intake` (`28f`) |
| **`PatientGender`** | `str` | `"F"`, `"M"` | `idx 7..8` (`OneHotEncoder`) | Unseen values zero-vector safe via `handle_unknown='ignore'` |
| **`ProviderSpecialty`** | `str` | `"Cardiology"`, `"General Practice"`, `"Neurology"`, `"Orthopedics"`, `"Pediatrics"` | `idx 9..13` (`OneHotEncoder`) | `5` canonical categories |
| **`PatientMaritalStatus`** | `str` | `"Divorced"`, `"Married"`, `"Single"`, `"Widowed"` | `idx 14..17` (`OneHotEncoder`) | `4` canonical categories |
| **`PatientEmploymentStatus`** | `str` | `"Employed"`, `"Retired"`, `"Student"`, `"Unemployed"` | `idx 18..21` (`OneHotEncoder`) | `4` canonical categories |
| **`ClaimType`** | `str` | `"Emergency"`, `"Inpatient"`, `"Outpatient"`, `"Routine"` | `idx 22..25` (`OneHotEncoder`) | `4` canonical categories |
| **`ClaimSubmissionMethod`** | `str` | `"Online"`, `"Paper"`, `"Phone"` | `idx 26..28` (`OneHotEncoder`) | `3` canonical categories |

### 2.2 Why `extra="forbid"` Rejections Return `HTTP 422`

Consider a client erroneously sending `"claimAmount"` (camelCase) or an extra legacy field `"ClaimStatus": "Approved"`:

* **Without `extra="forbid"` (Silent Bug Risk):** Pydantic defaults to `extra="ignore"`, stripping `"claimAmount"` silently and failing or falling back on default zero arrays.
* **With `extra="forbid"` (`claims.py`):** Immediately fails model validation prior to prediction, returning **`HTTP 422 Unprocessable Entity`** detailing `"Extra inputs are not permitted"`. In Track B, `submission/results/verification.json` explicitly verifies **`4` negative `422` rejection tests** to prove API boundary hygiene!

---

## 3. Model Routing Identifiers & Response Envelopes

### 3.1 Supported Model Keys Across Routes

| Endpoint Tier | Supported `model` / Key Strings | Notes |
| :--- | :--- | :--- |
| **`POST :8000/predict/ml`** (Classical) | `LogisticRegression`, `RandomForest`, `XGBoost`, `SVM`, `GradientBoosting` | Matches exact basenames under `Approch-1/models/*.joblib`; `"GradientBoosting"` invokes `HistGradientBoostingClassifier`! |
| **`POST :8000/predict/dl`** (Deep Learning) | `mlp`, `residualmlp`, `cnn1d`, `tabtransformer`, `autoencoder` | Loads `Approch-2-DL/models/<key>_best.pt` + thresholds from `results/*_metrics.json` |
| **`POST :3000/api/agentic/predict`** (Node) | Fixed `"AgenticEnsemble-v1"` | Returns inline rule score + `agent_id: "agent-chain-001"` + `feature_count: 9` |

### 3.2 Score Semantics & Calibration Flag Awareness (`ADR-05`)

Every consumer (TypeScript `frontend/src/types.ts`, React charts, SIU analysts) must handle prediction probability fields alongside explicit calibration flags established in `DECISIONS.md` (`ADR-05`):

1. **Classical ML (`classical_service.py`):** Executes `.predict_proba(X)[:, 1]`. Note `class_weight='balanced'` and `scale_pos_weight = 15.6667` distort true posterior log-odds upward. `SVM` uses internal Platt `.predict_proba()` rather than raw `.predict()` margins (`39` vs `51` test positives).
2. **Supervised DL (`dl_service.py`):** Evaluates `torch.sigmoid(logit)` against `0.5`. Logits reflect `BCEWithLogitsLoss(pos_weight=15.6667)` weighting ($\Delta z \approx +2.7515$).
3. **`Autoencoder` (`dl_service.py`):** Emits monotonic anomaly score `1 / (1 + exp(-(recon_error - threshold)*5 / max(threshold, 1e-8)))` centered strictly on `threshold = 0.12858837842941284` (`score_is_calibrated = False`).
4. **`AgenticEnsemble-v1` (`predict.ts`):** Clamped affine heuristic `min(max(0.01*amt + 0.005*age - 2.0, 0), 1)`.

---

## 4. Schema Divergence Between Track A ML/DL vs. Approach 3 Agentic

An essential portability subtlety documented across `shared/schemas.ts`, `claims.py`, and `predict.ts` is field naming & dimensionality mismatch between the trained ML/DL microservices and the prototype agentic pipeline:

| Dimension | ML & DL Services (`:8000` & `/api/ml/*`) | Agentic Prototype & Shadow Engine (`:8020` & `/api/agentic/predict`) |
| :--- | :--- | :--- |
| **Input Field Case** | **`PascalCase`** (`ClaimAmount`, `PatientAge`, `ClaimDate`, `Cluster`) | **`snake_case` / mapped** (`claim_amount`, `patient_age`, `feature_count=9`) |
| **Feature Vector Width** | `29` engineered dims (`28` in Track B `intake` ex-`Cluster`) | Labels `"feature_count": 9`; risk formula only evaluates `claim_amount` & `patient_age` |
| **Temporal Attribute** | Expects ISO date string `ClaimDate` -> expands Year/Month/DoW | Does not decompose `ClaimDate` in score rule |
| **Output Structure** | Risk score / probability + discrete fraud label + model metadata | Adds `4`-stage `audit_log` array + static `XAIExplainer` importance map + `12`-char SHA-256 `id` |

When wiring Approach 3 directly onto Tier 1 `ml-service` in future releases (see Volume `18`), Node `predict.ts` or an intermediate adapter layer must transform normalized camel/snake payloads into strict PascalCase `ClaimRequest` keys so Pydantic `extra="forbid"` accepts requests cleanly without returning HTTP `422`.

---

## 5. Express Health & Chat Endpoint Contracts (`:3000`)

### 5.1 Health Endpoint (`backend/node-api/src/routes/health.ts`, L1–7)

Inspected directly in source:

* Route: `GET /api/health` (mounted at `/api/health` inside `src/server.ts`)
* Response (`200 OK`): Exactly `{ "status": "ok" }`
* Consumed by `frontend/src/lib/useApiHealth.ts` (`30,000 ms` polling cadence via `checkApiHealth()` returning `"checking" | "online" | "offline"`).

### 5.2 Gemini Copilot Chat Endpoint (`backend/node-api/src/routes/chat.ts`, `47` lines)

* Route: `POST /api/chat`
* Expected JSON Body: Analyst text query (`message`) + optional serialized claim metadata & prediction snapshot (`context`) forwarded from `ChatBox.tsx`.
* Upstream Dispatch: Calls `@google/generative-ai` (`GoogleGenerativeAI`) configuring **`model: "gemini-3.8-flash"`** using `GEMINI_API_KEY` loaded from `../../Approach-3-Agentic-AI/.env`.

---

## 6. Executable PowerShell Contract Verification Suite

Use the PowerShell snippets below to validate live endpoint schema discipline across Tier 1 (`:8000`) and Tier 2 (`:3000`) directly:

```powershell
# 1. Verify Tier 2 Health & Tier 1 Readiness
Invoke-RestMethod -Uri "http://127.0.0.1:3000/api/health" -Method Get
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get

# 2. Construct Strict Valid PascalCase Claim Payload (High-Value Claim)
$validClaim = @{
    ClaimAmount             = 8950.50
    PatientAge              = 45
    PatientIncome           = 54000.0
    ClaimDate               = "2024-03-19"
    Cluster                 = 1
    PatientGender           = "F"
    ProviderSpecialty       = "Cardiology"
    PatientMaritalStatus    = "Married"
    PatientEmploymentStatus = "Employed"
    ClaimType               = "Inpatient"
    ClaimSubmissionMethod   = "Online"
} | ConvertTo-Json

# 3. Query Approach 1 Classical ML & Approach 2 Deep Learning through Express Gateway (:3000)
Invoke-RestMethod -Uri "http://127.0.0.1:3000/api/ml/predict/ml" -Method Post -ContentType "application/json" -Body $validClaim
Invoke-RestMethod -Uri "http://127.0.0.1:3000/api/ml/predict/dl" -Method Post -ContentType "application/json" -Body $validClaim

# 4. Negative Test: Verify FastAPI extra="forbid" Rejects Undeclared Field With HTTP 422
$badClaim = @{
    ClaimAmount = 8950.50
    UnknownJunkField = "should_reject_422"
} | ConvertTo-Json

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/ml" -Method Post -ContentType "application/json" -Body $badClaim
} catch {
    Write-Host "Confirmed Rejection Status:" $_.Exception.Response.StatusCode.value__
}
```
