# Python ML Service

FastAPI inference service for Mini-Project 2 supporting both Approach 1 (Classical ML) and Approach 2 (Deep Learning).

## Endpoints

```text
GET  /health
POST /predict/dl?model=<MLP|ResidualMLP|CNN1D|TabTransformer|Autoencoder>
POST /predict/ml?model=<RandomForest|XGBoost|GradientBoosting|SVM|LogisticRegression>
POST /predict  (defaults to DL MLP for backwards compatibility)
```

## Supported Models

- **Classical ML (`/predict/ml`):**
  - Uses Ganesh's frozen preprocessing pipeline from `Approch-1/data/processed/preprocessor.joblib`.
  - Loads models from `Approch-1/models/`: `RandomForest.joblib`, `XGBoost.joblib`, `GradientBoosting.joblib`, `SVM.joblib`, `LogisticRegression.joblib`.

- **Deep Learning (`/predict/dl`):**
  - Uses the frozen 29-feature contract and preprocessor.
  - Loads PyTorch models from `Approch-2-DL/models/`:
    - `MLP`: `mlp/mlp_best.pt`
    - `ResidualMLP`: `residualmlp/residualmlp_best.pt`
    - `CNN1D`: `cnn1d/cnn1d_best.pt`
    - `TabTransformer`: `tabtransformer/tabtransformer_best.pt`
    - `Autoencoder`: `autoencoder/autoencoder_best.pt` (anomaly detection via reconstruction error, threshold = 0.1286)

## How to Run

```powershell
cd E:\mini-project
.\Approch-2-DL\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir ml-service
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.
