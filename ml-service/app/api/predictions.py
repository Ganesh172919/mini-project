from fastapi import APIRouter

from app.schemas.claims import ClaimInput, PredictionResponse
from app.services.mlp_service import MLPInferenceService

router = APIRouter()
_service = MLPInferenceService()


@router.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimInput) -> PredictionResponse:
    return _service.predict(claim)
