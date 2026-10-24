"""Prediction API endpoints for both Deep Learning and Classical ML approaches."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.schemas.claims import ClaimInput, PredictionResponse
from app.services.classical_service import predict_classical, SUPPORTED_MODELS
from app.services.dl_service import predict_dl, DLModelName

router = APIRouter()

ClassicalModelName = Literal[
    "LogisticRegression", "RandomForest", "XGBoost", "SVM", "GradientBoosting"
]


# ─── Classical ML endpoint ───────────────────────────────────────────────────

@router.post("/predict/ml", response_model=PredictionResponse, tags=["classical-ml"])
def predict_ml_endpoint(
    claim: ClaimInput,
    model: ClassicalModelName = Query(default="RandomForest", description="Classical ML model to use"),
) -> PredictionResponse:
    """Run classical ML inference using Ganesh's trained scikit-learn models."""
    try:
        return predict_classical(claim, model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ─── Deep Learning endpoints ────────────────────────────────────────────────

@router.post("/predict/dl", response_model=PredictionResponse, tags=["deep-learning"])
def predict_dl_endpoint(
    claim: ClaimInput,
    model: DLModelName = Query(default="MLP", description="Approach 2 DL architecture to use"),
) -> PredictionResponse:
    """Run Approach 2 Deep Learning inference (MLP, ResidualMLP, CNN1D, TabTransformer, Autoencoder)."""
    try:
        return predict_dl(claim, model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/predict", response_model=PredictionResponse, tags=["deep-learning"])
def predict_default(
    claim: ClaimInput,
    model: DLModelName = Query(default="MLP", description="Approach 2 DL architecture to use"),
) -> PredictionResponse:
    """Default prediction endpoint (points to Deep Learning model, defaults to MLP)."""
    try:
        return predict_dl(claim, model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
