"""Classical ML inference service using Ganesh's fitted preprocessor and joblib models.

All five model artifacts are loaded once at startup from Approch-1/models/.
The preprocessor is from Approch-1/data/processed/preprocessor.joblib.
This module DOES NOT retrain or modify Ganesh's model files.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from app.schemas.claims import ClaimInput, PredictionResponse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
APPROCH_1_ROOT = PROJECT_ROOT / "Approch-1"
PREPROCESSOR_PATH = APPROCH_1_ROOT / "data" / "processed" / "preprocessor.joblib"
MODELS_DIR = APPROCH_1_ROOT / "models"

SUPPORTED_MODELS = {
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "SVM",
    "GradientBoosting",
}


@lru_cache(maxsize=1)
def _load_preprocessor() -> Any:
    if not PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    if not hasattr(preprocessor, "transform"):
        raise TypeError("Loaded artifact is not a scikit-learn transformer")
    return preprocessor


@lru_cache(maxsize=5)
def _load_model(model_name: str) -> Any:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name!r}. Choose from {sorted(SUPPORTED_MODELS)}")
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Model {model_name} requires package {exc.name!r} which is not installed in the ML service environment."
        ) from exc


def _frame_from_claim(claim: ClaimInput) -> pd.DataFrame:
    """Convert ClaimInput to the feature DataFrame expected by the preprocessor."""
    claim_data = claim.model_dump()
    claim_date: date = claim_data["ClaimDate"]
    row = {
        "ClaimAmount": claim_data["ClaimAmount"],
        "PatientAge": claim_data["PatientAge"],
        "PatientGender": claim_data["PatientGender"],
        "ProviderSpecialty": claim_data["ProviderSpecialty"],
        "PatientIncome": claim_data["PatientIncome"],
        "PatientMaritalStatus": claim_data["PatientMaritalStatus"],
        "PatientEmploymentStatus": claim_data["PatientEmploymentStatus"],
        "ClaimType": claim_data["ClaimType"],
        "ClaimSubmissionMethod": claim_data["ClaimSubmissionMethod"],
        "Cluster": claim_data["Cluster"],
        "ClaimYear": claim_date.year,
        "ClaimMonth": claim_date.month,
        "ClaimDayOfWeek": claim_date.weekday(),
    }
    return pd.DataFrame([row], columns=list(row.keys()))


def predict_classical(claim: ClaimInput, model_name: str) -> PredictionResponse:
    """Run inference for one claim using the named classical ML model."""
    preprocessor = _load_preprocessor()
    model = _load_model(model_name)
    raw_frame = _frame_from_claim(claim)
    features = np.asarray(preprocessor.transform(raw_frame), dtype=np.float32)
    if features.shape != (1, 29):
        raise ValueError(f"Preprocessor produced unexpected shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Preprocessor produced non-finite features")

    # Get fraud probability
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(features)[0, 1])
    else:
        # SVM with decision_function — convert via sigmoid
        score = float(expit(model.decision_function(features)[0]))

    return PredictionResponse(
        model_name=model_name,
        model_version=f"{model_name}.joblib",
        prediction="Fraud" if score >= 0.5 else "Legitimate",
        fraud_score=score,
        score_is_calibrated=False,
        feature_count=features.shape[1],
        feature_contract="Approch-1 frozen 29-feature leakage-excluded baseline",
        consumed_input_fields=[
            "ClaimAmount", "ClaimDate", "PatientAge", "PatientGender",
            "ProviderSpecialty", "PatientIncome", "PatientMaritalStatus",
            "PatientEmploymentStatus", "ClaimType", "ClaimSubmissionMethod", "Cluster",
        ],
        excluded_input_fields=[
            "ClaimID", "PatientID", "ProviderID", "DiagnosisCode",
            "ProcedureCode", "ClaimStatus", "ProviderLocation", "ClaimLegitimacy",
        ],
    )
