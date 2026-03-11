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


def make_preprocessor(
    num_cols: list[str], ohe_cols: list[str], te_cols: list[str]
) -> ColumnTransformer:
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
                    cols=te_cols,
                    smoothing=20,
                    handle_missing="value",
                    handle_unknown="value",
                ),
            )
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("ohe", ohe_pipe, ohe_cols),
            ("te", te_pipe, te_cols),
        ]
    )


def clean_and_engineer(
    df: pd.DataFrame,
    feature_cols: list[str],
    ohe_cols: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])

    trip_duration_min = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    pickup_hour = df["lpep_pickup_datetime"].dt.hour
    pickup_dayofweek = df["lpep_pickup_datetime"].dt.dayofweek

    df["pickup_month"] = df["lpep_pickup_datetime"].dt.month
    df["is_weekend"] = pickup_dayofweek.isin([5, 6]).astype(int)
    df["rush_hour"] = pickup_hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    df["route"] = (
        df["PULocationID"].astype("Int64").astype(str)
        + "_"
        + df["DOLocationID"].astype("Int64").astype(str)
    )

    mask = (
        trip_duration_min.between(1, 180)
        & df["fare_amount"].between(0.5, 500)
        & df["trip_distance"].between(0.01, 100)
        & df["passenger_count"].between(0, 8)
    )
    df = df.loc[mask].copy()

    df["trip_distance_log"] = np.log1p(df["trip_distance"])
    x = df[feature_cols].copy()
    y = np.log1p(df[target].copy())

    for col in ohe_cols:
        x[col] = x[col].astype(str)

    return x, y


def load_month(year: int, month: int) -> pd.DataFrame:
    url = BASE_URL.format(year=year, month=month)
    logging.info(f"Downloading {year}-{month:02d} ...")
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


def load_split_data(
    train_start_year: int,
    train_year: int,
    train_end_month: int,
    feature_cols: list[str],
    ohe_cols: list[str],
    target: str,
):
    if train_end_month < 1 or train_end_month > 12:
        raise ValueError("train_end_month must be between 1 and 12")

    test_year, test_month = next_month(train_year, train_end_month)

    logging.info(
        f"Training month: {train_start_year}.1 - {train_year}.{train_end_month}"
    )
    logging.info(f"Testing month: {test_year}.{test_month}")

    try:
        df_test_raw = load_month(test_year, test_month)
    except Exception as e:
        logging.info(
            f"Next-month target data {test_year}-{test_month:02d} unavailable. "
            f"Skipping run. Error: {e}"
        )
        return None

    train_pairs = list(
        iter_month_pairs(train_start_year, 1, train_year, train_end_month)
    )
    df_train_raw = load_month_pairs(train_pairs)

    x_train_full, y_train_full = clean_and_engineer(
        df_train_raw, feature_cols, ohe_cols, target
    )
    x_test, y_test = clean_and_engineer(df_test_raw, feature_cols, ohe_cols, target)

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


def run_retrain(
    data: dict,
    spec: dict,
    spec_path: Path,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    candidate_alias: str,
) -> int:
    target = spec["target"]
    num_cols = spec["num_cols"]
    ohe_cols = spec["ohe_cols"]
    te_cols = spec["te_cols"]
    selected_feature_names = spec["selected_feature_names"]

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    x_train_full = data["x_train_full"]
    y_train_full = data["y_train_full"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    preprocessor = make_preprocessor(num_cols, ohe_cols, te_cols)
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
    alpha_search_results = []

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    with mlflow.start_run(run_name="retrain_with_frozen_features") as run:
        for alpha in ridge_alphas:
            ridge = Ridge(alpha=alpha, random_state=42)
            ridge.fit(x_train_sel, y_train)
            val_rmse = rmse_from_log(y_val, ridge.predict(x_val_sel))
            alpha_search_results.append(
                {"alpha": float(alpha), "val_rmse": float(val_rmse)}
            )
            if val_rmse < best_ridge_val_rmse:
                best_ridge_val_rmse = val_rmse
                best_ridge_alpha = alpha

        final_preprocessor = make_preprocessor(num_cols, ohe_cols, te_cols)
        final_preprocessor.fit(x_train_full, y_train_full)
        final_feature_names = final_preprocessor.get_feature_names_out()
        final_name_to_idx = {n: i for i, n in enumerate(final_feature_names)}
        final_selected_indices = [final_name_to_idx[n] for n in selected_feature_names]

        final_inference_model = Pipeline(
            [
                ("preprocess", make_preprocessor(num_cols, ohe_cols, te_cols)),
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
        mlflow.log_param("alpha_search_iterations", len(alpha_search_results))
        mlflow.log_param("selected_feature_count", len(selected_feature_names))
        mlflow.log_param("best_ridge_alpha", float(best_ridge_alpha))
        mlflow.log_metric("best_ridge_val_rmse", best_ridge_val_rmse)
        mlflow.log_metric("final_test_rmse", final_test_rmse)

        mlflow.log_dict(alpha_search_results, "alpha_search_results.json")
        mlflow.log_dict(spec, "model_build_spec.json")
        mlflow.sklearn.log_model(final_inference_model, name="final_model")
        model_uri = f"runs:/{run.info.run_id}/final_model"

        mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        new_version = int(mv.version)

        client.set_registered_model_alias(
            registered_model_name, candidate_alias, new_version
        )
        logging.info(f"Set candidate alias '{candidate_alias}' -> v{new_version}")

        mlflow.log_param("registered_model_name", registered_model_name)
        mlflow.log_param("registered_model_version", new_version)
        mlflow.log_param("candidate_alias", candidate_alias)

    metrics = {
        "best_ridge_val_rmse": round(best_ridge_val_rmse, 3),
        "final_test_rmse": round(final_test_rmse, 3),
    }

    logging.info(f"Retrain complete")
    logging.info(f"Model URI: {model_uri}")
    logging.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    return 0


def load_spec(spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text())
    required_keys = {
        "target",
        "num_cols",
        "ohe_cols",
        "te_cols",
        "selected_feature_names",
    }
    missing_keys = sorted(required_keys - spec.keys())
    if missing_keys:
        raise ValueError(
            f"Spec file {spec_path} is missing required keys: {missing_keys}"
        )
    return spec


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
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--experiment-name", default="zoomcamp-model")
    parser.add_argument("--registered-model-name", default="nyc-taxi-ridge")
    parser.add_argument("--candidate-alias", default="candidate")
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

    return run_retrain(
        data=data,
        spec=spec,
        spec_path=spec_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        candidate_alias=args.candidate_alias,
    )


if __name__ == "__main__":
    raise SystemExit(main())
