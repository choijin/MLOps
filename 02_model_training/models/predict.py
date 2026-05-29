import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from joblib import load

from data.dataset_build import load_spec
from features.feature_engineering import records_to_features


DEFAULT_MODEL_ROOT = Path("artifacts/model_exports/champion")
DEFAULT_SPEC_PATH = Path("config/model_build_spec.json")


def resolve_model_path(model_path: Path) -> Path:
    """Resolve a direct or timestamped exported model path into model.joblib."""
    if model_path.is_file():
        return model_path

    direct_model = model_path / "model.joblib"
    if direct_model.is_file():
        return direct_model

    timestamped_models = sorted(
        [
            child / "model.joblib"
            for child in model_path.iterdir()
            if child.is_dir() and (child / "model.joblib").is_file()
        ],
        key=lambda path: path.parent.name,
    )
    if timestamped_models:
        return timestamped_models[-1]

    raise FileNotFoundError(f"No model.joblib found under {model_path}")


def load_model(model_path: Path):
    """Load a local exported model artifact from disk."""
    return load(resolve_model_path(model_path))


def predict_records(
    records: list[dict[str, Any]],
    model_path: Path = DEFAULT_MODEL_ROOT,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> list[float]:
    """Generate predictions from raw records using the latest local export by default."""
    spec = load_spec(spec_path)
    model = load_model(model_path)
    feature_cols = spec["num_cols"] + spec["ohe_cols"] + spec["te_cols"]
    features = records_to_features(records, feature_cols, spec["ohe_cols"])
    pred_log = model.predict(features)
    preds = np.expm1(pred_log)
    return [float(p) for p in preds]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local offline prediction from exported models."""
    parser = argparse.ArgumentParser(
        description="Load a serialized model artifact and generate predictions"
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_ROOT),
        help="Path to the exported model directory or model.joblib file",
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
    """Run local prediction from the command line and print JSON output."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    records = json.loads(args.input_json)
    if not isinstance(records, list):
        raise ValueError("--input-json must be a JSON array of records")

    predictions = predict_records(
        records=records,
        model_path=Path(args.model_path),
        spec_path=Path(args.spec_path),
    )
    logging.info(json.dumps({"predictions": predictions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
