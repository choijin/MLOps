from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes import router
from app.service import PredictionService


service = PredictionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup so requests can reuse it."""
    service.load()
    app.state.prediction_service = service
    yield


app = FastAPI(title="NYC Taxi Fare Prediction API", lifespan=lifespan)
app.include_router(router)
