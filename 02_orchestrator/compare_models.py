import argparse
import json
import logging
from datetime import date, datetime
from joblib import dump
from pathlib import Path
from typing import Any, Callable

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

try:
    from .dataset_build import choose_default_cutoff, load_spec, load_split_data
    from .evaluate import rmse_from_log
except ImportError:
    from dataset_build import choose_default_cutoff, load_spec, load_split_data
    from evaluate import rmse_from_log


def compare_and_promote_candidate(
    test_x: Any,
    test_y: Any,
    metric_fn: Callable[[Any, Any], float],
    tracking_uri: str,
    model_name: str,
    candidate_alias: str = "candidate",
    champion_alias: str = "champion",
    delete_candidate_on_promotion: bool = False,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    candidate_version = client.get_model_version_by_alias(
        model_name, candidate_alias
    ).version
    candidate_model = mlflow.sklearn.load_model(
        f"models:/{model_name}@{candidate_alias}"
    )
    candidate_score = float(metric_fn(test_y, candidate_model.predict(test_x)))

    champion_version = None
    champion_score = None

    try:
        champion_version = client.get_model_version_by_alias(
            model_name, champion_alias
        ).version
        champion_model = mlflow.sklearn.load_model(
            f"models:/{model_name}@{champion_alias}"
        )
        champion_score = float(metric_fn(test_y, champion_model.predict(test_x)))
    except Exception:
        logging.info(
            "No current champion found for model '%s'. Candidate v%s will be promoted.",
            model_name,
            candidate_version,
        )

    candidate_won = champion_score is None or candidate_score < champion_score
    champion_score_text = "N/A" if champion_score is None else f"{champion_score:.4f}"
    logging.info(
        "Model comparison complete. Champion RMSE: %s | Candidate RMSE: %.4f",
        champion_score_text,
        candidate_score,
    )

    if candidate_won:
        client.set_registered_model_alias(model_name, champion_alias, candidate_version)
        logging.info(
            "Promoted candidate v%s to alias '%s'.",
            candidate_version,
            champion_alias,
        )
        if delete_candidate_on_promotion:
            client.delete_registered_model_alias(model_name, candidate_alias)
            logging.info("Removed alias '%s' after promotion.", candidate_alias)
    else:
        logging.info(
            "Champion v%s remains active. Candidate v%s was not promoted.",
            champion_version,
            candidate_version,
        )

    return {
        "candidate_version": int(candidate_version),
        "candidate_rmse": candidate_score,
        "champion_version": (
            int(champion_version) if champion_version is not None else None
        ),
        "champion_rmse": champion_score,
        "candidate_won": candidate_won,
        "promoted_version": (
            int(candidate_version)
            if candidate_won
            else (int(champion_version) if champion_version is not None else None)
        ),
    }


def export_model_for_serving(
    tracking_uri: str,
    model_name: str,
    model_alias: str,
    export_dir: Path,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_version = client.get_model_version_by_alias(model_name, model_alias).version
    model = mlflow.sklearn.load_model(f"models:/{model_name}@{model_alias}")

    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "model.joblib"
    metadata_path = export_dir / "metadata.json"
    dump(model, export_path)
    metadata = {
        "model_name": model_name,
        "model_alias": model_alias,
        "model_version": int(model_version),
        "model_uri": f"models:/{model_name}@{model_alias}",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logging.info("Exported alias '%s' model to %s", model_alias, export_path)
    return metadata


def default_export_dir(model_alias: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("models") / model_alias / timestamp)


def run_model_promotion(
    data: dict,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    candidate_alias: str,
    champion_alias: str,
    delete_candidate_on_promotion: bool,
    export_dir: Path | None,
) -> int:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="promote_candidate_model") as run:
        promotion_result = compare_and_promote_candidate(
            test_x=data["x_test"],
            test_y=data["y_test"],
            metric_fn=rmse_from_log,
            tracking_uri=tracking_uri,
            model_name=registered_model_name,
            candidate_alias=candidate_alias,
            champion_alias=champion_alias,
            delete_candidate_on_promotion=delete_candidate_on_promotion,
        )

        mlflow.log_param("mode", "model_promotion")
        mlflow.log_param("train_start_year", data["train_start_year"])
        mlflow.log_param("train_year", data["train_year"])
        mlflow.log_param("train_end_month", data["train_end_month"])
        mlflow.log_param("test_year", data["test_year"])
        mlflow.log_param("test_month", data["test_month"])
        mlflow.log_param("registered_model_name", registered_model_name)
        mlflow.log_param("candidate_alias", candidate_alias)
        mlflow.log_param("champion_alias", champion_alias)
        mlflow.log_param("candidate_version", promotion_result["candidate_version"])
        mlflow.log_param("candidate_won", promotion_result["candidate_won"])
        if promotion_result["champion_version"] is not None:
            mlflow.log_param(
                "previous_champion_version", promotion_result["champion_version"]
            )
        if promotion_result["promoted_version"] is not None:
            mlflow.log_param("promoted_version", promotion_result["promoted_version"])

        mlflow.log_metric("candidate_alias_rmse", promotion_result["candidate_rmse"])
        if promotion_result["champion_rmse"] is not None:
            mlflow.log_metric("champion_alias_rmse", promotion_result["champion_rmse"])

        if export_dir is not None:
            export_metadata = export_model_for_serving(
                tracking_uri=tracking_uri,
                model_name=registered_model_name,
                model_alias=champion_alias,
                export_dir=export_dir,
            )
            mlflow.log_dict(export_metadata, "serving_export_metadata.json")

    metrics = {
        "candidate_alias_rmse": round(promotion_result["candidate_rmse"], 3),
    }
    if promotion_result["champion_rmse"] is not None:
        metrics["champion_alias_rmse"] = round(promotion_result["champion_rmse"], 3)

    logging.info("Promotion run complete")
    logging.info("Run ID: %s", run.info.run_id)
    logging.info("Metrics: %s", json.dumps(metrics, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    today = date.today()
    default_year, default_end = choose_default_cutoff(today)

    parser = argparse.ArgumentParser(
        description="Compare candidate model against champion and promote if better"
    )
    parser.add_argument("--train-start-year", type=int, default=2024)
    parser.add_argument("--train-year", type=int, default=default_year)
    parser.add_argument("--train-end-month", type=int, default=default_end)
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
        "--export-dir",
        default=default_export_dir("champion"),
        help="Optional local directory to export the active serving model",
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
        args.train_start_year,
        args.train_year,
        args.train_end_month,
        feature_cols,
        spec["ohe_cols"],
        spec["target"],
    )
    if data is None:
        return 0

    return run_model_promotion(
        data=data,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        candidate_alias=args.candidate_alias,
        champion_alias=args.champion_alias,
        delete_candidate_on_promotion=args.delete_candidate_on_promotion,
        export_dir=Path(args.export_dir) if args.export_dir else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
