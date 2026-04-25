import argparse
import logging
from pathlib import Path

try:
    from .dataset_build import load_spec, load_split_data
    from .train import run_train
except ImportError:
    from dataset_build import load_spec, load_split_data
    from train import run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deployment-stage training pipeline"
    )
    parser.add_argument("--train-start-year", type=int, default=2024)
    parser.add_argument("--train-year", type=int, required=True)
    parser.add_argument("--train-end-month", type=int, required=True)
    parser.add_argument("--spec-path", default="model_build_spec.json")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="zoomcamp-model")
    parser.add_argument("--registered-model-name", default="nyc-taxi-ridge")
    parser.add_argument("--candidate-alias", default="candidate")
    parser.add_argument("--export-dir", default="models")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    args = parse_args()
    spec_path = Path(args.spec_path)
    spec = load_spec(spec_path)
    feature_cols = spec["num_cols"] + spec["ohe_cols"] + spec["te_cols"]

    data = load_split_data(
        train_start_year=args.train_start_year,
        train_year=args.train_year,
        train_end_month=args.train_end_month,
        feature_cols=feature_cols,
        ohe_cols=spec["ohe_cols"],
        target=spec["target"],
    )
    if data is None:
        return 0

    return run_train(
        data=data,
        spec=spec,
        spec_path=spec_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        candidate_alias=args.candidate_alias,
        export_dir=Path(args.export_dir) if args.export_dir else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
