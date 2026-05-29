from typing import Any

import numpy as np
import pandas as pd

RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    if "lpep_pickup_datetime" in features.columns:
        pickup_ts = pd.to_datetime(features["lpep_pickup_datetime"])
        pickup_hour = pickup_ts.dt.hour
        pickup_dayofweek = pickup_ts.dt.dayofweek

        if "pickup_month" not in features.columns:
            features["pickup_month"] = pickup_ts.dt.month
        if "is_weekend" not in features.columns:
            features["is_weekend"] = pickup_dayofweek.isin([5, 6]).astype(int)
        if "rush_hour" not in features.columns:
            features["rush_hour"] = pickup_hour.isin(RUSH_HOURS).astype(int)

    if "route" not in features.columns and {
        "PULocationID",
        "DOLocationID",
    }.issubset(features.columns):
        features["route"] = (
            features["PULocationID"].astype("Int64").astype(str)
            + "_"
            + features["DOLocationID"].astype("Int64").astype(str)
        )

    if "trip_distance_log" not in features.columns and "trip_distance" in features.columns:
        features["trip_distance_log"] = np.log1p(features["trip_distance"])

    return features


def select_model_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    ohe_cols: list[str],
) -> pd.DataFrame:
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    x = df[feature_cols].copy()
    for col in ohe_cols:
        x[col] = x[col].astype(str)
    return x


def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    ohe_cols: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    features = build_features(df)

    if "lpep_dropoff_datetime" not in features.columns:
        raise ValueError("Training data is missing lpep_dropoff_datetime")

    pickup_ts = pd.to_datetime(features["lpep_pickup_datetime"])
    dropoff_ts = pd.to_datetime(features["lpep_dropoff_datetime"])
    trip_duration_min = (dropoff_ts - pickup_ts).dt.total_seconds() / 60

    mask = (
        trip_duration_min.between(1, 180)
        & features["fare_amount"].between(0.5, 500)
        & features["trip_distance"].between(0.01, 100)
        & features["passenger_count"].between(0, 8)
    )
    filtered = features.loc[mask].copy()

    x = select_model_features(filtered, feature_cols, ohe_cols)
    y = np.log1p(filtered[target].copy())
    return x, y


def records_to_features(
    records: list[dict[str, Any]],
    feature_cols: list[str],
    ohe_cols: list[str],
) -> pd.DataFrame:
    df = build_features(pd.DataFrame(records))
    return select_model_features(df, feature_cols, ohe_cols)
