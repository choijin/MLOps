import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np

from src.data.dataset_build import load_spec
from src.features.feature_engineering import records_to_features


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = BASE_DIR / "config" / "model_build_spec.json"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
DEFAULT_MODEL_ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "final_model")
DEFAULT_RUN_ID = os.getenv("RUN_ID")


def prepare_features(
    records: list[dict[str, Any]],
    spec_path: Path = DEFAULT_SPEC_PATH,
):
    """Build model-ready features from raw request records."""
    spec = load_spec(spec_path)
    feature_cols = spec["num_cols"] + spec["ohe_cols"] + spec["te_cols"]
    return records_to_features(records, feature_cols, spec["ohe_cols"])


def predict_records(
    records: list[dict[str, Any]],
    model=None,
    run_id: str | None = DEFAULT_RUN_ID,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    model_artifact_path: str = DEFAULT_MODEL_ARTIFACT_PATH,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> list[float]:
    """Predict fares from raw records using a loaded or run-resolved MLflow model."""
    if model is None:
        if not run_id:
            raise ValueError("A run_id is required when no preloaded model is provided.")
        mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_artifact_path}")
    features = prepare_features(records, spec_path=spec_path)
    pred_log = model.predict(features)
    preds = np.expm1(pred_log)
    return [round(float(p), 3) for p in preds]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch prediction."""
    parser = argparse.ArgumentParser(
        description="Load an MLflow model artifact and generate predictions"
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="MLflow run_id that produced the logged model artifact",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI used to load the model artifact for the given run_id",
    )
    parser.add_argument(
        "--model-artifact-path",
        default=DEFAULT_MODEL_ARTIFACT_PATH,
        help="Logged MLflow model name to resolve for the given run_id",
    )
    parser.add_argument(
        "--spec-path",
        default=str(DEFAULT_SPEC_PATH),
        help="Path to the model build spec JSON file",
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON array of feature records to score",
    )
    return parser.parse_args()


def main() -> int:
    """Run prediction from the command line and print JSON output."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    records = json.loads(args.input_json)
    if not isinstance(records, list):
        raise ValueError("--input-json must be a JSON array of records")

    predictions = predict_records(
        records=records,
        run_id=args.run_id,
        tracking_uri=args.tracking_uri,
        model_artifact_path=args.model_artifact_path,
        spec_path=Path(args.spec_path),
    )
    logging.info(
        "Predictions: %s",
        json.dumps({"predictions": predictions}, indent=2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
