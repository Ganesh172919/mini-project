# Project Decisions

1. `Approch-1/` is owned by Ganesh and is not modified by Deep Learning work.
2. Deep Learning work is isolated under `Approch-2-DL/`.
3. A dedicated virtual environment is used at `Approch-2-DL/.venv`.
4. The initial DL baseline uses the existing 29-feature preprocessing contract for comparison with the classical ML baseline.
5. MLP is the first DL model.
6. PR-AUC is the primary evaluation metric because the dataset has approximately 6% fraud cases.
7. React → Node.js → FastAPI is the planned application architecture.
8. Agentic AI is a separate team component owned by Varshith.
9. The FastAPI prediction service reuses the fitted preprocessing transformer from `Approch-1/data/processed/preprocessor.joblib` read-only, while the MLP checkpoint and frozen feature-name contract remain under `Approch-2-DL/`.
10. The prediction API reports an uncalibrated `fraud_score`, not a calibrated probability.
