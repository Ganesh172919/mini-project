from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from app.schemas.claims import ClaimInput, PredictionResponse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DL_ROOT = PROJECT_ROOT / "Approch-2-DL"
PREPROCESSOR_PATH = PROJECT_ROOT / "Approch-1" / "data" / "processed" / "preprocessor.joblib"
FEATURE_NAMES_PATH = DL_ROOT / "results" / "processed" / "feature_names.txt"
CHECKPOINT_PATH = DL_ROOT / "models" / "mlp" / "mlp_best.pt"

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
ENGINEERED_FEATURES = ["ClaimYear", "ClaimMonth", "ClaimDayOfWeek"]
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


class MLPInferenceService:
    def __init__(self) -> None:
        self.preprocessor = self._load_preprocessor()
        self.feature_names = self._load_feature_names()
        self.model = self._load_model()

    @staticmethod
    def _require_file(path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"Required inference artifact is missing: {path}")
        return path

    def _load_preprocessor(self) -> Any:
        preprocessor = joblib.load(self._require_file(PREPROCESSOR_PATH))
        if not hasattr(preprocessor, "transform") or not hasattr(preprocessor, "get_feature_names_out"):
            raise TypeError(f"Preprocessor is incompatible: {PREPROCESSOR_PATH}")
        return preprocessor

    def _load_feature_names(self) -> list[str]:
        names = self._require_file(FEATURE_NAMES_PATH).read_text(encoding="utf-8").splitlines()
        if len(names) != 29:
            raise ValueError(f"Expected 29 feature names, found {len(names)}")
        produced_names = [name.split("__", 1)[-1] for name in self.preprocessor.get_feature_names_out()]
        if produced_names != names:
            raise ValueError(
                "Preprocessor feature ordering is incompatible with the frozen feature_names.txt contract"
            )
        return names

    def _load_model(self) -> torch.nn.Module:
        sys.path.insert(0, str(DL_ROOT / "scripts"))
        from dl_common import MLP

        checkpoint = torch.load(self._require_file(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
        input_dim = int(checkpoint.get("input_dim", -1))
        if input_dim != len(self.feature_names):
            raise ValueError(f"Checkpoint expects {input_dim} features; contract provides {len(self.feature_names)}")
        model = MLP(input_dim=input_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model_version = CHECKPOINT_PATH.name
        return model

    @staticmethod
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
        return pd.DataFrame([row], columns=[
            "ClaimAmount",
            "PatientAge",
            "PatientGender",
            "ProviderSpecialty",
            "PatientIncome",
            "PatientMaritalStatus",
            "PatientEmploymentStatus",
            "ClaimType",
            "ClaimSubmissionMethod",
            "Cluster",
            "ClaimYear",
            "ClaimMonth",
            "ClaimDayOfWeek",
        ])

    def predict(self, claim: ClaimInput) -> PredictionResponse:
        raw_frame = self._frame_from_claim(claim)
        features = np.asarray(self.preprocessor.transform(raw_frame), dtype=np.float32)
        if features.shape != (1, len(self.feature_names)):
            raise ValueError(f"Preprocessor produced incompatible shape: {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("Preprocessor produced non-finite model features")
        with torch.inference_mode():
            logit = self.model(torch.from_numpy(features))
            score = float(torch.sigmoid(logit).item())
        return PredictionResponse(
            model_name="MLP",
            model_version=self.model_version,
            prediction="Fraud" if score >= 0.5 else "Legitimate",
            fraud_score=score,
            score_is_calibrated=False,
            feature_count=features.shape[1],
            feature_contract="Approch-1 frozen 29-feature leakage-excluded baseline",
            consumed_input_fields=CONSUMED_INPUT_FIELDS,
            excluded_input_fields=EXCLUDED_INPUT_FIELDS,
        )
