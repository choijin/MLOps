from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class SparseColumnSelector(BaseEstimator, TransformerMixin):
    """Select a fixed set of transformed sparse feature columns by index."""
    def __init__(self, indices: list[int]):
        self.indices = indices

    def fit(self, x, y=None):
        """Return the selector unchanged because fitting is not required."""
        return self

    def transform(self, x):
        """Subset the transformed matrix to the configured feature indices."""
        return x[:, self.indices]


def make_preprocessor(
    num_cols: list[str], ohe_cols: list[str], te_cols: list[str]
) -> ColumnTransformer:
    """Build the shared preprocessing pipeline used for training and inference."""
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "ohe",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ohe_cols,
            ),
            (
                "te",
                Pipeline(
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
                ),
                te_cols,
            ),
        ]
    )


def select_feature_indices(
    preprocessor: ColumnTransformer,
    selected_feature_names: list[str],
) -> list[int]:
    """Map selected transformed feature names back to their column indices."""
    feature_names = preprocessor.get_feature_names_out()
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    missing = [name for name in selected_feature_names if name not in name_to_idx]
    if missing:
        raise ValueError(
            f"Selected features missing in current preprocessor output: {missing}"
        )
    return [name_to_idx[name] for name in selected_feature_names]


def build_inference_model(
    num_cols: list[str],
    ohe_cols: list[str],
    te_cols: list[str],
    selected_indices: list[int],
    alpha: float,
) -> Pipeline:
    """Build the final preprocessing, selection, and Ridge regression pipeline."""
    return Pipeline(
        [
            ("preprocess", make_preprocessor(num_cols, ohe_cols, te_cols)),
            ("select", SparseColumnSelector(selected_indices)),
            ("reg", Ridge(alpha=alpha, random_state=42)),
        ]
    )
