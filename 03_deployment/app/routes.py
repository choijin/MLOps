from fastapi import APIRouter, Request

from app.schemas import HealthResponse, PredictionRequest, PredictionResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple readiness response."""
    return HealthResponse(status="ok")


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, app_request: Request) -> PredictionResponse:
    """Score incoming trip records and return fare predictions."""
    service = app_request.app.state.prediction_service
    predictions = service.predict(request.records)
    return PredictionResponse(predictions=predictions)
