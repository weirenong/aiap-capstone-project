"""Preprocessing for the ads "None" classification problem.

The target is ``clicked_ad``: 1 = the user clicks, 0 = none of the ad categories is
clicked. The 0 class is the "none" / background class (98% of the data), so every
transform here has to protect the 2% positive class rather than optimise overall
accuracy.

Design rules, all of them consequences of that framing:

1. Row filtering happens before the split. Dropping rows inside a Pipeline would
   desynchronise X and y, so eligibility rules live in ``filter_eligible_rows`` and are
   applied once, to the whole frame, before anything is learned.
2. Nothing that learns from data is fitted before the split. Imputation values, IQR
   fences, scaler statistics and the category vocabulary all come from ``fit`` on the
   training fold only. Fitting on the full frame is the classic way to inflate a score.
3. No resampling here. SMOTE and friends belong inside the CV loop, so
   ``build_resampled_model`` wraps them in an imbalanced-learn pipeline instead of
   exposing them as a preprocessing step you might call early by accident.
4. Unseen categories at inference time are encoded as an all-zero block
   (``handle_unknown="ignore"``). That is the right behaviour for a None problem: a
   category the model has never seen is evidence for "none of my known classes", not a
   reason to crash or to guess.
5. Constant ``none`` predictions are the failure we are guarding against, so no step
   here drops the minority class, caps it, or reorders the label encoding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # keeps both `python main.py` and `python -m src.preprocessing` working
    from src.feature_engineering import PriorHistoryFeatures
except ImportError:  # pragma: no cover - when src/ itself is on sys.path
    from feature_engineering import PriorHistoryFeatures

# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------

TARGET = "clicked_ad"
NONE_LABEL = 0
POSITIVE_LABEL = 1

NUMERIC_FEATURES = ("user_age", "previous_clicks_on_similar_ads")
CATEGORICAL_FEATURES = (
    "gender",
    "device_type",
    "time_of_day",
    "ad_placement_position",
    "ad_category",
)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MIN_AGE = 18
RANDOM_STATE = 42

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "synthetic_data_243.csv"


# --------------------------------------------------------------------------------------
# Loading and row-level cleaning
# --------------------------------------------------------------------------------------


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and check the schema."""
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise if the frame cannot be used as-is.

    Cheap to call and it catches the failure mode that wastes the most time: a column
    silently missing or renamed between one run and the next.
    """
    missing = [c for c in FEATURES + (TARGET,) if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    if df[TARGET].isna().any():
        raise ValueError("target column contains nulls")

    unexpected = set(df[TARGET].unique()) - {NONE_LABEL, POSITIVE_LABEL}
    if unexpected:
        raise ValueError(f"target must be binary 0/1, found {sorted(unexpected)}")

    if not pd.api.types.is_numeric_dtype(df["user_age"]):
        raise TypeError("user_age must be numeric")

    if df.empty:
        raise ValueError("dataset is empty")


def filter_eligible_rows(
    df: pd.DataFrame, min_age: int = MIN_AGE
) -> tuple[pd.DataFrame, dict]:
    """Drop rows that should never reach the model. Returns (frame, report).

    The EDA found 45 rows with a user younger than 18 (minimum 6.99). For an ad-click
    model that is a legal/ethical problem and a distribution problem, so those rows are
    removed here rather than learned around. Keep this before ``train_test_split``: a
    filter applied after the split changes the test population.
    """
    under_age = df["user_age"] < min_age
    duplicate_rows = df.duplicated()
    keep = ~(under_age | duplicate_rows)

    report = {
        "rows_in": int(len(df)),
        "under_min_age": int(under_age.sum()),
        "duplicate_rows": int(duplicate_rows.sum()),
        "rows_out": int(keep.sum()),
        "positives_in": int(df[TARGET].sum()),
        "positives_out": int(df.loc[keep, TARGET].sum()),
        "positives_dropped": int(df.loc[~keep, TARGET].sum()),
    }
    return df.loc[keep].reset_index(drop=True), report


# --------------------------------------------------------------------------------------
# Transformers
# --------------------------------------------------------------------------------------


def _as_matrix(X) -> tuple[np.ndarray, list[str] | None]:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(), list(X.columns)
    return np.asarray(X), None


class IQRClipper(TransformerMixin, BaseEstimator):
    """Clip numeric columns to their training IQR fences.

    Outliers are compressed rather than dropped: with 20 positives, deleting rows is not
    an option, and the tail beyond the fence carries little information at this sample
    size. Fences are learned in ``fit`` and stored per column, so a test row above the
    training fence gets clipped rather than redefining the boundary.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        """Fit on training data only; every learned statistic belongs to the fold it saw."""
        matrix, names = _as_matrix(X)
        q1 = np.nanpercentile(matrix, 25, axis=0)
        q3 = np.nanpercentile(matrix, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - self.factor * iqr
        upper = q3 + self.factor * iqr
        # Two cases must not be clipped:
        #  - an all-null column has NaN quantiles
        #  - a zero-IQR column (a binary flag, or any constant) has no tail to trim, and
        #    clipping it to q1..q3 would silently destroy the feature. That bug turned
        #    the low-history flag into a column of zeros on the first run.
        no_spread = np.isnan(lower) | (iqr == 0)
        self.lower_ = np.where(no_spread, -np.inf, lower)
        self.upper_ = np.where(no_spread, np.inf, upper)
        self.feature_names_in_ = np.asarray(names) if names is not None else None
        self.n_features_in_ = matrix.shape[1]
        return self

    def transform(self, X):
        """Apply the fitted transform and return the produced columns."""
        matrix, _ = _as_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} columns, got {matrix.shape[1]}"
            )
        return np.clip(matrix, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        """Names of the columns this transformer produces."""
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        if input_features is None:
            raise ValueError("input_features required when fitted on a numpy array")
        return np.asarray(input_features, dtype=object)


# --------------------------------------------------------------------------------------
# Composed pipeline
# --------------------------------------------------------------------------------------


def build_preprocessor(
    *,
    engineer_features: bool = True,
    clip_outliers: bool = True,
    scale_numeric: bool = True,
    numeric_imputer: str = "median",
    categorical_imputer: str = "most_frequent",
    add_missing_indicator: bool = False,
    iqr_factor: float = 1.5,
) -> ColumnTransformer:
    """Column-wise preprocessing. Engineer -> impute -> clip -> scale, in that order.

    ``add_missing_indicator`` is off by default because the current file has no nulls;
    turning it on is harmless and gives the model a "was this value absent" signal if
    later data does have gaps.
    """
    numeric_steps = []
    if engineer_features:
        numeric_steps.append(("engineer", PriorHistoryFeatures()))
    numeric_steps.append(
        ("impute", SimpleImputer(strategy=numeric_imputer, add_indicator=add_missing_indicator))
    )
    if clip_outliers:
        numeric_steps.append(("clip", IQRClipper(factor=iqr_factor)))
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy=categorical_imputer)),
        (
            "encode",
            # handle_unknown="ignore": an unseen category becomes an all-zero block,
            # i.e. evidence against every known category rather than a crash.
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            ("categorical", Pipeline(categorical_steps), list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model(
    estimator,
    *,
    engineer_features: bool = True,
    clip_outliers: bool = True,
    scale_numeric: bool = True,
    iqr_factor: float = 1.5,
) -> Pipeline:
    """Preprocessor + classifier as one fitted object.

    Everything the model needs travels with it, so fitting on a CV fold and predicting
    on the held-out fold cannot accidentally reuse full-data statistics.
    """
    return Pipeline(
        [
            (
                "preprocess",
                build_preprocessor(
                    engineer_features=engineer_features,
                    clip_outliers=clip_outliers,
                    scale_numeric=scale_numeric,
                    iqr_factor=iqr_factor,
                ),
            ),
            ("model", estimator),
        ]
    )


def build_resampled_model(estimator, sampler, **preprocess_kwargs) -> Pipeline:
    """Like ``build_model`` but with a sampler applied inside each training fold.

    Deliberately a separate function so that resampling cannot be dropped into the
    preprocessing chain where it would run before the split. The sampler only ever sees
    the fold it is fitted on, which is the whole point.
    """
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "imbalanced-learn is required for resampling; install it or use build_model "
            "with class_weight='balanced'"
        ) from exc

    return ImbPipeline(
        [
            ("preprocess", build_preprocessor(**preprocess_kwargs)),
            ("sample", sampler),
            ("model", estimator),
        ]
    )


# --------------------------------------------------------------------------------------
# Splitting helpers
# --------------------------------------------------------------------------------------


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target, with the none label left untouched."""
    return df[list(FEATURES)].copy(), df[TARGET].astype(int).copy()


def make_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
):
    """Stratified split. Stratification is non-negotiable at a 2% positive rate."""
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def make_open_set_split(
    df: pd.DataFrame,
    holdout_category: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Known-class train/test plus a separate set from a class never seen in training.

    Used to test the None problem properly: the model is fitted without
    ``holdout_category`` and should flag those rows as none-of-the-above instead of
    assigning them a known class.

    Warning: this dataset has 20 positives in total, and a held-out category holds a
    handful of them (Travel 6, Gaming 7). Use it as a sanity check on behaviour, not as
    a source of headline metrics.
    """
    if holdout_category not in set(df["ad_category"]):
        raise ValueError(f"unknown category: {holdout_category}")

    is_unknown = df["ad_category"] == holdout_category
    known_df = df.loc[~is_unknown].reset_index(drop=True)
    unknown_df = df.loc[is_unknown].reset_index(drop=True)

    X_train, X_test, y_train, y_test = make_train_test_split(
        known_df, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, unknown_df


# --------------------------------------------------------------------------------------
# Small helpers used by the smoke test in main.py
# --------------------------------------------------------------------------------------


def positive_rate(y) -> float:
    """Share of rows that are not the none class."""
    y = np.asarray(y)
    return float((y == POSITIVE_LABEL).mean())


def transform_frame(preprocessor, X: pd.DataFrame) -> pd.DataFrame:
    """Transform to a labelled DataFrame, for inspection rather than modelling."""
    matrix = preprocessor.transform(X)
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame(matrix, columns=names, index=X.index)


__all__ = [
    "TARGET",
    "NONE_LABEL",
    "POSITIVE_LABEL",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "FEATURES",
    "MIN_AGE",
    "RANDOM_STATE",
    "DEFAULT_DATA_PATH",
    "IQRClipper",
    "PriorHistoryFeatures",
    "load_data",
    "validate_schema",
    "filter_eligible_rows",
    "build_preprocessor",
    "build_model",
    "build_resampled_model",
    "split_features_target",
    "make_train_test_split",
    "make_open_set_split",
    "positive_rate",
    "transform_frame",
]
