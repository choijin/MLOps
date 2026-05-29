from pathlib import Path

from src.models.predict import (
    DEFAULT_LOCAL_MODEL_PATH,
    DEFAULT_MODEL_ARTIFACT_PATH,
    DEFAULT_RUN_ID,
    DEFAULT_SPEC_PATH,
    DEFAULT_TRACKING_URI,
    load_local_model,
    load_mlflow_model,
    predict_records,
)


class PredictionService:
    """Load the model once and reuse it for API prediction requests."""

    def __init__(
        self,
        local_model_path: Path = DEFAULT_LOCAL_MODEL_PATH,
        run_id: str | None = DEFAULT_RUN_ID,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        model_artifact_path: str = DEFAULT_MODEL_ARTIFACT_PATH,
        spec_path: Path = DEFAULT_SPEC_PATH,
    ) -> None:
        """Initialize the service with MLflow run selection and spec locations."""
        self.local_model_path = local_model_path
        self.run_id = run_id
        self.tracking_uri = tracking_uri
        self.model_artifact_path = model_artifact_path
        self.spec_path = spec_path
        self.model = None

    def load(self) -> None:
        """Load the requested MLflow run or bundled model into memory."""
        if self.model is None:
            if self.run_id:
                self.model = load_mlflow_model(
                    run_id=self.run_id,
                    tracking_uri=self.tracking_uri,
                    model_artifact_path=self.model_artifact_path,
                )
            elif self.local_model_path.is_file():
                self.model = load_local_model(self.local_model_path)
            else:
                raise ValueError(
                    "No bundled model found and RUN_ID was not provided for MLflow loading."
                )

    def predict(self, records: list[dict]) -> list[float]:
        """Score records using the cached model instance."""
        self.load()
        return predict_records(
            records=records,
            model=self.model,
            spec_path=self.spec_path,
        )
