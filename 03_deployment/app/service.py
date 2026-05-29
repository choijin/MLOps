from pathlib import Path

import mlflow
import mlflow.sklearn

from src.models.predict import (
    DEFAULT_MODEL_ARTIFACT_PATH,
    DEFAULT_RUN_ID,
    DEFAULT_SPEC_PATH,
    DEFAULT_TRACKING_URI,
    predict_records,
)


class PredictionService:
    """Load the model once and reuse it for API prediction requests."""

    def __init__(
        self,
        run_id: str | None = DEFAULT_RUN_ID,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        model_artifact_path: str = DEFAULT_MODEL_ARTIFACT_PATH,
        spec_path: Path = DEFAULT_SPEC_PATH,
    ) -> None:
        """Initialize the service with MLflow run selection and spec locations."""
        self.run_id = run_id
        self.tracking_uri = tracking_uri
        self.model_artifact_path = model_artifact_path
        self.spec_path = spec_path
        self.model = None

    def load(self) -> None:
        """Load the MLflow run artifact into memory if it is not loaded yet."""
        if self.model is None:
            if not self.run_id:
                raise ValueError(
                    "RUN_ID must be set before starting the prediction service."
                )
            mlflow.set_tracking_uri(self.tracking_uri)
            self.model = mlflow.sklearn.load_model(
                f"runs:/{self.run_id}/{self.model_artifact_path}"
            )

    def predict(self, records: list[dict]) -> list[float]:
        """Score records using the cached model instance."""
        self.load()
        return predict_records(
            records=records,
            model=self.model,
            spec_path=self.spec_path,
        )
