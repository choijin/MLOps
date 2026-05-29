from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for scoring one or more trip records."""

    records: list[dict[str, Any]] = Field(
        ..., description="Raw trip records to score."
    )


class PredictionResponse(BaseModel):
    """Response body containing predicted fare values."""

    predictions: list[float]


class HealthResponse(BaseModel):
    """Simple health response for uptime checks."""

    status: str
