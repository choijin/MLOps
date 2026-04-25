import argparse
import logging
from datetime import datetime
from pathlib import Path

try:
    from .compare_models import run_model_promotion
    from .dataset_build import load_spec, load_split_data
    from .train import run_train
except ImportError:
    from compare_models import run_model_promotion
    from dataset_build import load_spec, load_split_data
    from train import run_train


def default_export_dir(model_alias: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("models") / model_alias / timestamp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full training pipeline: train candidate, then compare and promote"
    )
    parser.add_argument("--train-start-year", type=int, default=2024)
    parser.add_argument("--train-year", type=int, required=True)
    parser.add_argument("--train-end-month", type=int, required=True)
    parser.add_argument("--spec-path", default="model_build_spec.json")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="zoomcamp-model")
    parser.add_argument("--registered-model-name", default="nyc-taxi-ridge")
    parser.add_argument("--candidate-alias", default="candidate")
    parser.add_argument("--champion-alias", default="champion")
    parser.add_argument(
        "--delete-candidate-on-promotion",
        action="store_true",
        help="Remove the candidate alias after it becomes champion",
    )
    parser.add_argument(
        "--candidate-export-dir",
        default=default_export_dir("candidate"),
        help="Optional local directory to export the candidate model",
    )
    parser.add_argument(
        "--champion-export-dir",
        default=default_export_dir("champion"),
        help="Optional local directory to export the active champion model",
    )
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

    train_result = run_train(
        data=data,
        spec=spec,
        spec_path=spec_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        candidate_alias=args.candidate_alias,
        export_dir=(
            Path(args.candidate_export_dir) if args.candidate_export_dir else None
        ),
    )
    if train_result != 0:
        return train_result

    return run_model_promotion(
        data=data,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        candidate_alias=args.candidate_alias,
        champion_alias=args.champion_alias,
        delete_candidate_on_promotion=args.delete_candidate_on_promotion,
        export_dir=(
            Path(args.champion_export_dir) if args.champion_export_dir else None
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
