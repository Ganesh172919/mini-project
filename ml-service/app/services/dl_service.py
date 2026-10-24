"""Multi-model Deep Learning inference service for Approach 2.

Supports all five trained architectures:
- MLP
- ResidualMLP
- CNN1D
- TabTransformer
- Autoencoder (anomaly detection via reconstruction error)

All models share the frozen 29-feature contract and preprocessor.
Artifacts are cached in memory after first load.
"""
from __future__ import annotations

import sys
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import torch

from app.schemas.claims import ClaimInput, PredictionResponse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DL_ROOT = PROJECT_ROOT / "Approch-2-DL"
SCRIPTS_DIR = DL_ROOT / "scripts"
MODELS_DIR = DL_ROOT / "models"
PREPROCESSOR_PATH = PROJECT_ROOT / "Approch-1" / "data" / "processed" / "preprocessor.joblib"
FEATURE_NAMES_PATH = DL_ROOT / "results" / "processed" / "feature_names.txt"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dl_common import MLP  # noqa: E402
from submission_models import ResidualMLP, CNN1D  # noqa: E402
from dl_advanced_models import TabTransformer, Autoencoder  # noqa: E402

DLModelName = Literal["MLP", "ResidualMLP", "CNN1D", "TabTransformer", "Autoencoder"]

CHECKPOINT_MAP: dict[str, Path] = {
    "MLP": MODELS_DIR / "mlp" / "mlp_best.pt",
    "ResidualMLP": MODELS_DIR / "residualmlp" / "residualmlp_best.pt",
    "CNN1D": MODELS_DIR / "cnn1d" / "cnn1d_best.pt",
    "TabTransformer": MODELS_DIR / "tabtransformer" / "tabtransformer_best.pt",
    "Autoencoder": MODELS_DIR / "autoencoder" / "autoencoder_best.pt",
}

CONSUMED_INPUT_FIELDS = [
    "ClaimAmount",
    "ClaimDate",
    "PatientAge",
    "PatientGender",
    "ProviderSpecialty",
    "PatientIncome",
    "PatientMaritalStatus",
    "PatientEmploymentStatus",
    "ClaimType",
    "ClaimSubmissionMethod",
    "Cluster",
]
EXCLUDED_INPUT_FIELDS = [
    "ClaimID",
    "PatientID",
    "ProviderID",
    "DiagnosisCode",
    "ProcedureCode",
    "ClaimStatus",
    "ProviderLocation",
    "ClaimLegitimacy",
]


@lru_cache(maxsize=1)
def _load_preprocessor() -> Any:
    if not PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError(f"Preprocessor artifact missing: {PREPROCESSOR_PATH}")
    pre = joblib.load(PREPROCESSOR_PATH)
    if not hasattr(pre, "transform"):
        raise TypeError("Loaded preprocessor has no transform method")
    return pre


@lru_cache(maxsize=1)
def _load_feature_names() -> list[str]:
    if not FEATURE_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Feature names artifact missing: {FEATURE_NAMES_PATH}")
    names = FEATURE_NAMES_PATH.read_text(encoding="utf-8").splitlines()
    if len(names) != 29:
        raise ValueError(f"Expected 29 feature names, found {len(names)}")
    return names


@lru_cache(maxsize=5)
def _load_model(model_name: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    if model_name not in CHECKPOINT_MAP:
        raise ValueError(f"Unknown DL model: {model_name!r}. Supported: {list(CHECKPOINT_MAP)}")
    ckpt_path = CHECKPOINT_MAP[model_name]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint.get("input_dim", 29))

    if model_name == "MLP":
        model = MLP(input_dim=input_dim)
    elif model_name == "ResidualMLP":
        model = ResidualMLP(input_dim=input_dim)
    elif model_name == "CNN1D":
        model = CNN1D(input_dim=input_dim)
    elif model_name == "TabTransformer":
        model = TabTransformer(input_dim=input_dim)
    elif model_name == "Autoencoder":
        model = Autoencoder(input_dim=input_dim)
        model.threshold = float(checkpoint.get("threshold", 0.1))
    else:
        raise ValueError(f"Unhandled architecture: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _frame_from_claim(claim: ClaimInput) -> pd.DataFrame:
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


def predict_dl(claim: ClaimInput, model_name: DLModelName = "MLP") -> PredictionResponse:
    """Run inference using the specified Approach 2 DL model."""
    preprocessor = _load_preprocessor()
    feature_names = _load_feature_names()
    model, checkpoint = _load_model(model_name)

    raw_frame = _frame_from_claim(claim)
    features = np.asarray(preprocessor.transform(raw_frame), dtype=np.float32)
    if features.shape != (1, len(feature_names)):
        raise ValueError(f"Preprocessor output shape mismatch: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Preprocessor produced non-finite features")

    feat_tensor = torch.from_numpy(features)

    with torch.inference_mode():
        if model_name == "Autoencoder":
            recon_error = float(model.reconstruction_error(feat_tensor).item())
            threshold = float(getattr(model, "threshold", 0.1))
            # Normalise to [0, 1] using sigmoid around threshold
            score = float(1.0 / (1.0 + np.exp(-(recon_error - threshold) * 5.0 / max(threshold, 1e-8))))
            is_fraud = recon_error >= threshold
            score_calibrated = False
        else:
            logit = model(feat_tensor)
            score = float(torch.sigmoid(logit).item())
            is_fraud = score >= 0.5
            score_calibrated = False

    return PredictionResponse(
        model_name=model_name,
        model_version=CHECKPOINT_MAP[model_name].name,
        prediction="Fraud" if is_fraud else "Legitimate",
        fraud_score=score,
        score_is_calibrated=score_calibrated,
        feature_count=features.shape[1],
        feature_contract="Approch-1 frozen 29-feature leakage-excluded baseline",
        consumed_input_fields=CONSUMED_INPUT_FIELDS,
        excluded_input_fields=EXCLUDED_INPUT_FIELDS,
    )
