import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterable

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"

FEATURE_COLS = [
    "VendorID",
    "passenger_count",
    "trip_distance_log",
    "RatecodeID",
    "trip_type",
    "store_and_fwd_flag",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "rush_hour",
    "trip_duration_min_log",
    "route",
]
TARGET = "fare_amount"
OHE_COLS = ["VendorID", "RatecodeID", "trip_type", "store_and_fwd_flag"]
TE_COLS = ["route"]


class SparseColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, indices: list[int]):
        self.indices = indices

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[:, self.indices]


def rmse_from_log(y_true_log: pd.Series, y_pred_log: np.ndarray) -> float:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_preprocessor(kept_num_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ohe_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    te_pipe = Pipeline(
        [
            (
                "te",
                TargetEncoder(
                    cols=TE_COLS,
                    smoothing=20,
                    handle_missing="value",
                    handle_unknown="value",
                ),
            )
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipe, kept_num_cols),
            ("ohe", ohe_pipe, OHE_COLS),
            ("te", te_pipe, TE_COLS),
        ]
    )


def clean_and_engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])

    df["trip_duration_min"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["lpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["lpep_pickup_datetime"].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    df["route"] = (
        df["PULocationID"].astype("Int64").astype(str)
        + "_"
        + df["DOLocationID"].astype("Int64").astype(str)
    )

    mask = (
        df["trip_duration_min"].between(1, 180)
        & df["fare_amount"].between(0.5, 500)
        & df["trip_distance"].between(0.01, 100)
        & df["passenger_count"].between(0, 8)
    )
    df = df.loc[mask].copy()

    df["trip_distance_log"] = np.log1p(df["trip_distance"])
    df["trip_duration_min_log"] = np.log1p(df["trip_duration_min"])

    x = df[FEATURE_COLS].copy()
    y = np.log1p(df[TARGET].copy())

    for col in OHE_COLS:
        x[col] = x[col].astype(str)

    return x, y


def load_month(year: int, month: int) -> pd.DataFrame:
    url = BASE_URL.format(year=year, month=month)
    logging.info("Downloading %d-%02d ...", year, month)
    return pd.read_parquet(url)


def iter_month_pairs(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def load_month_pairs(pairs: Iterable[tuple[int, int]]) -> pd.DataFrame:
    frames = [load_month(y, m) for y, m in pairs]
    return pd.concat(frames, ignore_index=True)


def next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def choose_default_cutoff(today: date) -> tuple[int, int]:
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1


def load_split_data(train_start_year: int, train_year: int, train_end_month: int):
    if train_end_month < 1 or train_end_month > 12:
        raise ValueError("train_end_month must be between 1 and 12")

    test_year, test_month = next_month(train_year, train_end_month)

    logging.info(
        "Training month: %d.1 - %d.%d", train_start_year, train_year, train_end_month
    )
    logging.info("Testing month: %d.%d", test_year, test_month)

    try:
        df_test_raw = load_month(test_year, test_month)
    except Exception as e:
        logging.info(
            "Next-month target data %d-%02d unavailable. Skipping run. Error: %s",
            test_year,
            test_month,
            e,
        )
        return None

    train_pairs = list(
        iter_month_pairs(train_start_year, 1, train_year, train_end_month)
    )
    df_train_raw = load_month_pairs(train_pairs)

    x_train_full, y_train_full = clean_and_engineer(df_train_raw)
    x_test, y_test = clean_and_engineer(df_test_raw)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.3, random_state=42
    )

    return {
        "train_start_year": train_start_year,
        "train_year": train_year,
        "train_end_month": train_end_month,
        "test_year": test_year,
        "test_month": test_month,
        "x_train_full": x_train_full,
        "y_train_full": y_train_full,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def get_active_model_rmse(client: MlflowClient, model_name: str, active_alias: str):
    try:
        mv = client.get_model_version_by_alias(model_name, active_alias)
    except Exception:
        return None, None

    run = client.get_run(mv.run_id)
    if not run:
        return mv, None
    return mv, run.data.metrics.get("final_test_rmse")


def promote_candidate_if_better(
    client: MlflowClient,
    registered_model_name: str,
    active_alias: str,
    candidate_version: int,
    candidate_rmse: float,
) -> tuple[bool, float | None]:
    promoted = False
    active_mv, active_rmse = get_active_model_rmse(
        client, registered_model_name, active_alias
    )

    if active_mv is None or active_rmse is None or candidate_rmse < float(active_rmse):
        client.set_registered_model_alias(
            registered_model_name, active_alias, candidate_version
        )
        logging.info("Candidate v%s promoted to '%s'.", candidate_version, active_alias)
        promoted = True
    else:
        logging.info(
            "Candidate v%s not promoted. active_rmse=%.3f, candidate_rmse=%.3f",
            candidate_version,
            float(active_rmse),
            candidate_rmse,
        )

    return promoted, active_rmse


def run_retrain(
    data: dict,
    spec_path: Path,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    active_alias: str,
    candidate_alias: str,
) -> int:
    spec = json.loads(spec_path.read_text())

    if "kept_num_cols" not in spec or "selected_feature_names" not in spec:
        raise ValueError(
            f"Spec file {spec_path} must contain kept_num_cols and selected_feature_names"
        )

    kept_num_cols = spec["kept_num_cols"]
    selected_feature_names = spec["selected_feature_names"]

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    x_train_full = data["x_train_full"]
    y_train_full = data["y_train_full"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    preprocessor = make_preprocessor(kept_num_cols)
    x_train_tx = preprocessor.fit_transform(x_train, y_train)
    feature_names = preprocessor.get_feature_names_out()

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    missing = [n for n in selected_feature_names if n not in name_to_idx]
    if missing:
        raise ValueError(
            f"Selected features missing in current preprocessor output: {missing}"
        )

    selected_indices = [name_to_idx[n] for n in selected_feature_names]
    x_train_sel = x_train_tx[:, selected_indices]

    x_val_tx = preprocessor.transform(x_val)
    x_val_sel = x_val_tx[:, selected_indices]

    ridge_alphas = np.logspace(-4, 4, 50)
    best_ridge_alpha = None
    best_ridge_val_rmse = float("inf")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    with mlflow.start_run(run_name="retrain_with_frozen_features") as run:
        for alpha in ridge_alphas:
            with mlflow.start_run(run_name=f"ridge_alpha_{alpha:.6f}", nested=True):
                ridge = Ridge(alpha=alpha, random_state=42)
                ridge.fit(x_train_sel, y_train)
                val_rmse = rmse_from_log(y_val, ridge.predict(x_val_sel))
                mlflow.log_param("alpha", float(alpha))
                mlflow.log_metric("val_rmse", val_rmse)
                if val_rmse < best_ridge_val_rmse:
                    best_ridge_val_rmse = val_rmse
                    best_ridge_alpha = alpha

        final_preprocessor = make_preprocessor(kept_num_cols)
        x_train_full_tx = final_preprocessor.fit_transform(x_train_full, y_train_full)
        final_feature_names = final_preprocessor.get_feature_names_out()
        final_name_to_idx = {n: i for i, n in enumerate(final_feature_names)}
        final_selected_indices = [final_name_to_idx[n] for n in selected_feature_names]

        final_inference_model = Pipeline(
            [
                ("preprocess", make_preprocessor(kept_num_cols)),
                ("select", SparseColumnSelector(final_selected_indices)),
                ("reg", Ridge(alpha=best_ridge_alpha, random_state=42)),
            ]
        )
        final_inference_model.fit(x_train_full, y_train_full)

        final_test_rmse = rmse_from_log(y_test, final_inference_model.predict(x_test))

        mlflow.log_param("mode", "retrain")
        mlflow.log_param("train_start_year", data["train_start_year"])
        mlflow.log_param("train_year", data["train_year"])
        mlflow.log_param("train_end_month", data["train_end_month"])
        mlflow.log_param("test_year", data["test_year"])
        mlflow.log_param("test_month", data["test_month"])
        mlflow.log_param("spec_path", str(spec_path))
        mlflow.log_param("selected_feature_count", len(selected_feature_names))
        mlflow.log_param("best_ridge_alpha", float(best_ridge_alpha))
        mlflow.log_metric("best_ridge_val_rmse", best_ridge_val_rmse)
        mlflow.log_metric("final_test_rmse", final_test_rmse)

        mlflow.log_dict(
            {"selected_feature_names": selected_feature_names},
            "selected_features.json",
        )
        mlflow.sklearn.log_model(final_inference_model, name="final_model")
        model_uri = f"runs:/{run.info.run_id}/final_model"

        mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        new_version = int(mv.version)

        client.set_registered_model_alias(
            registered_model_name, candidate_alias, new_version
        )
        logging.info("Set candidate alias '%s' -> v%s", candidate_alias, new_version)

        promoted, active_rmse = promote_candidate_if_better(
            client=client,
            registered_model_name=registered_model_name,
            active_alias=active_alias,
            candidate_version=new_version,
            candidate_rmse=final_test_rmse,
        )

        mlflow.log_param("registered_model_name", registered_model_name)
        mlflow.log_param("registered_model_version", new_version)
        mlflow.log_param("active_alias", active_alias)
        mlflow.log_param("candidate_alias", candidate_alias)
        mlflow.log_param("promoted_to_active", promoted)
        if active_rmse is not None:
            mlflow.log_metric("active_model_test_rmse", float(active_rmse))

    metrics = {
        "best_ridge_val_rmse": round(best_ridge_val_rmse, 3),
        "final_test_rmse": round(final_test_rmse, 3),
    }

    logging.info("Retrain complete")
    logging.info("Model URI: %s", model_uri)
    logging.info("Metrics: %s", json.dumps(metrics, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    today = date.today()
    default_year, default_end = choose_default_cutoff(today)

    parser = argparse.ArgumentParser(
        description="Retrain monthly model with frozen variables from notebook build spec"
    )
    parser.add_argument("--train-start-year", type=int, default=2024)
    parser.add_argument("--train-year", type=int, default=default_year)
    parser.add_argument("--train-end-month", type=int, default=default_end)
    parser.add_argument("--spec-path", default="model_build_spec.json")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="zoomcamp-model")
    parser.add_argument("--registered-model-name", default="nyc-taxi-ridge")
    parser.add_argument("--active-alias", default="active")
    parser.add_argument("--candidate-alias", default="candidate")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    args = parse_args()
    data = load_split_data(args.train_start_year, args.train_year, args.train_end_month)
    if data is None:
        return 0

    return run_retrain(
        data=data,
        spec_path=Path(args.spec_path),
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        active_alias=args.active_alias,
        candidate_alias=args.candidate_alias,
    )


if __name__ == "__main__":
    raise SystemExit(main())
