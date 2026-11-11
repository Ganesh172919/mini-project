# Volume 16 — Deployment Runbooks, Windows DevOps & Diagnostic Troubleshooting

> **Document Classification:** Operations Manual, Environment Setup & Incident Remediation Playbook  
> **Target Platform:** Windows OS (`PowerShell`), Python `3.11.x` `.venv`, Node.js `v18+` / `v20+` + Vite, FastAPI/Uvicorn, Chromium/Node verifier.  
> **Primary Ground-Truth Sources:** [`README.md`](../../README.md), [`Approch-1/documentation/06_setup_and_usage.md`](../../Approch-1/documentation/06_setup_and_usage.md), [`Approch-2-DL/README.md`](../../Approch-2-DL/README.md), [`ml-service/README.md`](../../ml-service/README.md), [`backend/node-api/README.md`](../../backend/node-api/README.md), [`Approach-3-Agentic-AI/docs/README.md`](../../Approach-3-Agentic-AI/docs/README.md), and [`submission/START_DEMO.ps1`](../../submission/START_DEMO.ps1).

---

## 1. Environment Topology & Virtual Environment Isolation Map (`ADR-02`)

To guarantee dependency hygiene across heterogeneous Python ML libraries (`scikit-learn 1.9.1`, `xgboost`, `torch`), web runtime dependencies (`fastapi`, `uvicorn`, `pydantic`), Node gateway packages (`express`, `@google/generative-ai`), and Track B export/verification tools, avoid installing packages into the global Windows Python interpreter:

| Subsystem Directory | Interpreter / Package Manager | Local Environment Path | Key Port Bound (if Service) |
| :--- | :--- | :--- | :---: |
| `Approch-1/` | Python `3.11.2` + `pip` | `Approch-1\.venv\` | Batch scripts only (needs `.\project` junction) |
| `Approch-2-DL/` | Python + PyTorch (`requirements.txt`) | `Approch-2-DL\.venv\` | Batch scripts only (consumes `Approch-1` arrays) |
| `ml-service/` | Python + FastAPI + Uvicorn | `ml-service\.venv\` | **`:8000`** (`127.0.0.1:8000`) |
| `Approach-3-Agentic-AI/` | Python + FastAPI + Uvicorn | `Approach-3-Agentic-AI\.venv\` | **`:8020`** (`127.0.0.1:8020`, standalone) |
| `backend/node-api/` | Node.js + `npm` + `tsx` | `backend\node-api\node_modules\` | **`:3000`** (`127.0.0.1:3000`) |
| `frontend/` | Node.js + `npm` + Vite | `frontend\node_modules\` | **`:5173`** (`localhost:5173`) |
| `submission/` | Python `.venv` + Node (`check_browser.mjs`) | `submission\.venv\` | **`:8080`** (`127.0.0.1:8080`) |

---

## 2. Full-Stack Launch Playbooks (Windows PowerShell)

### 2.1 Mode A: Track A Full-Stack Web Platform (`3` Concurrent Terminals)

Open **three dedicated PowerShell windows** from workspace root `c:\Users\RAVIPRAKASH\Downloads\mini-project`:

```powershell
# ================= TERMINAL 1: Tier 1 Python ML/DL Service (:8000) =================
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\ml-service
if (-not (Test-Path .\.venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000

# ================= TERMINAL 2: Tier 2 Express Node Gateway (:3000) =================
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\backend\node-api
# Optional: Ensure Approach-3-Agentic-AI/.env defines GEMINI_API_KEY=... for /api/chat
npm install
npm run dev

# ================= TERMINAL 3: Tier 3 React + Vite SPA (:5173) ===================
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\frontend
npm install
npm run dev
```

Verify readiness order strictly **bottom-up (`:8000 → :3000 → :5173`)**:
1. Confirm `http://127.0.0.1:8000/health` reports loaded Classical + DL artifacts first.
2. Confirm `http://127.0.0.1:3000/api/health` returns `{ "status": "ok" }`.
3. Navigate to `http://localhost:5173` in a modern desktop browser.

### 2.2 Mode B: Track B Zero-Backend Pure-JS Demo (`:8080`)

When presenting offline or running deterministic `intake` (`RandomForest`, threshold `0.4167`) pure-JS inference without Tier 1/2 servers:

```powershell
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\submission
.\START_DEMO.ps1
# Opens / serves static verified bundle on http://127.0.0.1:8080 via serve.py
```

### 2.3 Mode C: Approach 1 Training Re-Execution (Windows Junction Protocol)

Because [`Approch-1/documentation/06_setup_and_usage.md`](../../Approch-1/documentation/06_setup_and_usage.md) notes hardcoded relative paths prefixed with `project/`, create an NTFS junction before invoking `preprocess.py` or `train_all_models.py`:

```powershell
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\Approch-1
if (-not (Test-Path .\project)) {
    New-Item -ItemType Junction -Path ".\project" -Target "."
}
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/preprocess.py
python scripts/train_all_models.py
python scripts/comparison_plots.py
python scripts/error_analysis.py
```

### 2.4 Mode D: Launching Standalone `:8020` Prototype (`Approach-3-Agentic-AI/`)

Never execute `python run_agentic.py` directly (`include_router(predict_fraud)` defect). Start cleanly via ASGI entrypoint `app.main:app`:

```powershell
cd c:\Users\RAVIPRAKASH\Downloads\mini-project\Approach-3-Agentic-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8020
```

---

## 3. Environment Secret Provisioning (`GEMINI_API_KEY`)

The conversational analyst assistant in [`backend/node-api/src/routes/chat.ts`](../../backend/node-api/src/routes/chat.ts) reads `.env` explicitly from the sibling prototype directory rather than `backend/node-api/.env`:

1. Create file `c:\Users\RAVIPRAKASH\Downloads\mini-project\Approach-3-Agentic-AI\.env`.
2. Add your API credential cleanly without quotes or trailing spaces:
   ```env
   GEMINI_API_KEY=your_google_ai_studio_api_key_here
   ```
3. Restart Node Express (`npm run dev` in `backend/node-api/`) so `dotenv` reloads environment variables before instantiating `GoogleGenerativeAI` with model **`gemini-3.8-flash`**.
4. Ensure `.gitignore` protects `.env` files from accidental source commits before pushing changes.

---

## 4. `15`-Symptom Master Diagnostic Troubleshooting Matrix

| # | Observed Failure Symptom | Exact Root Cause Verified in Codebase | Exact Verified Remediation |
| :---: | :--- | :--- | :--- |
| **`01`** | `Approch-1` scripts throw `FileNotFoundError: 'project/data/raw/...'` | Hardcoded `project/` relative path inside `preprocess.py` & `train_all_models.py`. | Create Windows junction: `New-Item -ItemType Junction -Path '.\project' -Target '.'` inside `Approch-1\`. |
| **`02`** | `ClassicalMLPage` predicts `1%` (`0.01`) fraud risk even on `\$9,500` claims | Tier 2 `:3000` or Tier 1 `:8000` is unreachable; UI falls back silently to `ClassicalMLPage.tsx` L92–98 formula (`\$47k+` activation!). | Start `:8000` FastAPI + `:3000` Node gateway and resubmit claim. |
| **`03`** | `AgenticPage` scores every normal claim (`\$400`–`\$9,000`) at `100%` (`1.00`) | Active `predict.ts` (`:3000`) stub formula `0.01*amt + 0.005*age - 2.0` saturates at ceiling `1.0` once `amt >= \$260–\$300`. | Documented stub behavior (`Vol. 05`); wire `risk-scorer` stage to `:8000/predict/ml` in `predict.ts`. |
| **`04`** | `POST :3000/api/chat` fails immediately in `ChatBox` | Missing `GEMINI_API_KEY` or missing `Approach-3-Agentic-AI/.env` file referenced by `node-api/src/routes/chat.ts`. | Populate `Approach-3-Agentic-AI/.env` and restart Node `:3000`. |
| **`05`** | `python Approach-3-Agentic-AI/run_agentic.py` fails on launch | `run_agentic.py` erroneously passes view function `predict_fraud` into `app.include_router(predict_fraud)`. | Start using `uvicorn app.main:app --port 8020` instead of `run_agentic.py`. |
| **`06`** | Routes on `:8020` `Approach-3-Agentic-AI` `predict.py` missing or import fails | `predict.py` imports invalid `BackgroundTask` (singular) and redeclares `router = APIRouter(...)` on L6 & L30. | Repair `BackgroundTasks` import and delete duplicate `router = ...` assignment on Line 30. |
| **`07`** | `POST :8000/predict/ml` returns `HTTP 422` ("Extra inputs are not permitted") | `ClaimRequest` (`claims.py`) sets `extra="forbid"`; payload used `camelCase` or injected unexpected keys like `ClaimStatus`. | Send exact `11` PascalCase domain attributes listed in `Vol. 12`. |
| **`08`** | Offline `SVM.joblib` F1 (`83.52%`) differs from `ml-service` live `SVM` (`91.14%`) | `model_results.json` logged `.predict()` (`51` flagged), whereas `classical_service.py` invokes Platt `.predict_proba()` (`39` flagged). | Expected Platt calibration asymmetry (`legacy_artifact_audit.json`); Track B resolves via `expit(decision_function)`. |
| **`09`** | `ml-service` crashes on startup citing missing `.joblib` or `.pt` file | `Approch-1/models/*.joblib` or `Approch-2-DL/models/*_best.pt` missing/moved from sibling path. | Run `preprocess.py`/`train_all_models.py` and `train_*.py` or restore clean git artifacts. |
| **`10`** | Edits inside `frontend/src/agentic/pages/AgenticDashboard.tsx` don't show on `:5173` | `App.tsx` mounts `frontend/src/pages/AgenticPage.tsx` directly — `AgenticDashboard.tsx` & `ClaimInputForm.tsx` are unmounted. | Edit `frontend/src/pages/AgenticPage.tsx` or wire `AgenticDashboard` cleanly into `App.tsx`. |
| **`11`** | `ChatBox` icon/panel disappears when browsing Classical ML or DL tabs | `App.tsx` conditionally renders `<ChatBox />` strictly when `activeTab === "agentic"`. | Navigate back to `"Agentic AI"` tab (`activeTab === "agentic"`), or lift mount in `App.tsx`. |
| **`12`** | Changes to `Approch-1/scripts/create_notebooks.py` fail with `KeyError` / invalid syntax | `create_notebooks.py` leaves literal `{class_def}` unformatted inside string block. | Author notebooks interactively or fix `{class_def}` string interpolation cleanly before execution. |
| **`13`** | `Approch-2-DL/scripts/train_supervised.py` fails loading `.npy` | Missing `processed/X_*.npy` files. | Copy/run `Approch-1/scripts/preprocess.py` first so `.npy` splits exist cleanly. |
| **`14`** | `Autoencoder` flags legitimate claim or misses `\$8,500` synthetic fraud | `Autoencoder` achieves `5.23%` PR-AUC empirically (`tau = 0.1286`); legitimate variance masks frauds. | Intended scientific finding (`Vol. 04`, `Vol. 10`) — use `CNN1D` / `TabTransformer` / Classical ML instead. |
| **`15`** | Track B `ResidualMLP` takes `42` epochs instead of Track A's `20` | `submission/scripts/run_pipeline.py` activates `torch.use_deterministic_algorithms(True)`, altering batch op reduction order. | Expected cross-track reproducibility effect (`95.67%` vs `95.61%` PR-AUC, `Vol. 07`). |
