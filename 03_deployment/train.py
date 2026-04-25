import argparse
import json
import logging
from datetime import date
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from joblib import dump
from mlflow.tracking import MlflowClient
from sklearn.linear_model import Ridge

try:
    from .evaluate import rmse_from_log
    from .preprocessor import (
        build_inference_model,
        make_preprocessor,
        select_feature_indices,
    )
    from .dataset_build import choose_default_cutoff, load_spec, load_split_data
except ImportError:
    from evaluate import rmse_from_log
    from preprocessor import (
        build_inference_model,
        make_preprocessor,
        select_feature_indices,
    )
    from dataset_build import choose_default_cutoff, load_spec, load_split_data


def export_model_artifact(
    model,
    export_dir: Path,
    metadata: dict,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    dump(model, export_dir / "model.joblib")
    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logging.info("Exported candidate model to %s", export_dir / "model.joblib")


def choose_best_ridge_alpha(
    x_train,
    y_train,
    x_val,
    y_val,
) -> tuple[float, float, list[dict[str, float]]]:
    best_alpha = None
    best_val_rmse = float("inf")
    search_results = []

    for alpha in np.logspace(-4, 4, 50):
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(x_train, y_train)
        val_rmse = rmse_from_log(y_val, ridge.predict(x_val))
        search_results.append({"alpha": float(alpha), "val_rmse": float(val_rmse)})
        if val_rmse < best_val_rmse:
            best_val_rmse = float(val_rmse)
            best_alpha = float(alpha)

    return best_alpha, best_val_rmse, search_results


def run_train(
    data: dict,
    spec: dict,
    spec_path: Path,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    candidate_alias: str,
    export_dir: Path | None,
) -> int:
    num_cols = spec["num_cols"]
    ohe_cols = spec["ohe_cols"]
    te_cols = spec["te_cols"]
    selected_feature_names = spec["selected_feature_names"]

    preprocessor = make_preprocessor(num_cols, ohe_cols, te_cols)
    x_train_tx = preprocessor.fit_transform(data["x_train"], data["y_train"])
    selected_indices = select_feature_indices(preprocessor, selected_feature_names)
    x_val_tx = preprocessor.transform(data["x_val"])

    best_alpha, best_val_rmse, alpha_search_results = choose_best_ridge_alpha(
        x_train=x_train_tx[:, selected_indices],
        y_train=data["y_train"],
        x_val=x_val_tx[:, selected_indices],
        y_val=data["y_val"],
    )

    final_preprocessor = make_preprocessor(num_cols, ohe_cols, te_cols)
    final_preprocessor.fit(data["x_train_full"], data["y_train_full"])
    final_selected_indices = select_feature_indices(
        final_preprocessor, selected_feature_names
    )

    final_model = build_inference_model(
        num_cols=num_cols,
        ohe_cols=ohe_cols,
        te_cols=te_cols,
        selected_indices=final_selected_indices,
        alpha=best_alpha,
    )
    final_model.fit(data["x_train_full"], data["y_train_full"])
    final_test_rmse = rmse_from_log(data["y_test"], final_model.predict(data["x_test"]))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    with mlflow.start_run(run_name="retrain_with_frozen_features") as run:
        mlflow.log_param("mode", "retrain")
        mlflow.log_param("train_start_year", data["train_start_year"])
        mlflow.log_param("train_year", data["train_year"])
        mlflow.log_param("train_end_month", data["train_end_month"])
        mlflow.log_param("test_year", data["test_year"])
        mlflow.log_param("test_month", data["test_month"])
        mlflow.log_param("spec_path", str(spec_path))
        mlflow.log_param("alpha_search_iterations", len(alpha_search_results))
        mlflow.log_param("selected_feature_count", len(selected_feature_names))
        mlflow.log_param("best_ridge_alpha", best_alpha)
        mlflow.log_param("registered_model_name", registered_model_name)
        mlflow.log_metric("best_ridge_val_rmse", best_val_rmse)
        mlflow.log_metric("final_test_rmse", final_test_rmse)

        mlflow.log_dict(alpha_search_results, "alpha_search_results.json")
        mlflow.log_dict(spec, "model_build_spec.json")
        mlflow.sklearn.log_model(final_model, name="final_model")
        model_uri = f"runs:/{run.info.run_id}/final_model"

        model_version = int(
            mlflow.register_model(model_uri=model_uri, name=registered_model_name).version
        )
        client.set_registered_model_alias(
            registered_model_name, candidate_alias, model_version
        )
        mlflow.log_param("registered_model_version", model_version)
        mlflow.log_param("candidate_alias", candidate_alias)

        if export_dir is not None:
            export_model_artifact(
                model=final_model,
                export_dir=export_dir,
                metadata={
                    "model_name": registered_model_name,
                    "model_version": model_version,
                    "model_alias": candidate_alias,
                    "train_start_year": data["train_start_year"],
                    "train_year": data["train_year"],
                    "train_end_month": data["train_end_month"],
                    "test_year": data["test_year"],
                    "test_month": data["test_month"],
                    "spec_path": str(spec_path),
                },
            )

    logging.info("Retrain complete")
    logging.info("Model URI: %s", model_uri)
    logging.info(
        "Metrics: %s",
        json.dumps(
            {
                "best_ridge_val_rmse": round(best_val_rmse, 3),
                "final_test_rmse": round(final_test_rmse, 3),
            },
            indent=2,
        ),
    )
    return 0


def parse_args() -> argparse.Namespace:
    default_year, default_end = choose_default_cutoff(date.today())
    parser = argparse.ArgumentParser(
        description="Train a candidate model and register it in MLflow"
    )
    parser.add_argument("--train-start-year", type=int, default=2024)
    parser.add_argument("--train-year", type=int, default=default_year)
    parser.add_argument("--train-end-month", type=int, default=default_end)
    parser.add_argument("--spec-path", default="model_build_spec.json")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="zoomcamp-model")
    parser.add_argument("--registered-model-name", default="nyc-taxi-ridge")
    parser.add_argument("--candidate-alias", default="candidate")
    parser.add_argument(
        "--export-dir",
        default="models",
        help="Optional local directory to export the trained model",
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
