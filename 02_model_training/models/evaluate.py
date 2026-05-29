import numpy as np
import pandas as pd


def rmse_from_log(y_true_log: pd.Series, y_pred_log: np.ndarray) -> float:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))
