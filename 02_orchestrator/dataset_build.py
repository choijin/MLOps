import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .feature_engineering import prepare_training_data
except ImportError:
    from feature_engineering import prepare_training_data


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"


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


def load_month(year: int, month: int) -> pd.DataFrame:
    url = BASE_URL.format(year=year, month=month)
    logging.info("Downloading %s-%02d ...", year, month)
    return pd.read_parquet(url)


def iter_month_pairs(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> Iterable[tuple[int, int]]:
    year, month = start_year, start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        yield year, month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def load_month_pairs(pairs: Iterable[tuple[int, int]]) -> pd.DataFrame:
    frames = [load_month(year, month) for year, month in pairs]
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
) -> dict | None:
    if not 1 <= train_end_month <= 12:
        raise ValueError("train_end_month must be between 1 and 12")

    test_year, test_month = next_month(train_year, train_end_month)

    logging.info(
        "Training month: %s.1 - %s.%s",
        train_start_year,
        train_year,
        train_end_month,
    )
    logging.info("Testing month: %s.%s", test_year, test_month)

    try:
        df_test_raw = load_month(test_year, test_month)
    except Exception as exc:
        logging.info(
            "Next-month target data %s-%02d unavailable. Skipping run. Error: %s",
            test_year,
            test_month,
            exc,
        )
        return None

    train_pairs = list(
        iter_month_pairs(train_start_year, 1, train_year, train_end_month)
    )
    df_train_raw = load_month_pairs(train_pairs)

    x_train_full, y_train_full = prepare_training_data(
        df_train_raw, feature_cols, ohe_cols, target
    )
    x_test, y_test = prepare_training_data(
        df_test_raw, feature_cols, ohe_cols, target
    )

    train_idx, val_idx = _train_val_indices(len(x_train_full))
    x_train = x_train_full.iloc[train_idx].copy()
    y_train = y_train_full.iloc[train_idx].copy()
    x_val = x_train_full.iloc[val_idx].copy()
    y_val = y_train_full.iloc[val_idx].copy()

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


def _train_val_indices(
    n_rows: int, val_fraction: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=42)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    val_size = int(round(n_rows * val_fraction))
    val_idx = np.sort(indices[:val_size])
    train_idx = np.sort(indices[val_size:])
    return train_idx, val_idx
