from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.predictions import router as predictions_router

app = FastAPI(title="Health Fraud ML Service", version="0.1.0")
app.include_router(health_router)
app.include_router(predictions_router)
