"""Feature creation and feature selection for the ads "None" classification problem.

This module owns two jobs:

1. **Creating features.** Domain-motivated transforms that add information the raw
   columns don't carry directly. They are plain scikit-learn transformers, so they fit
   inside a Pipeline and learn only from the fold they are given.
2. **Selecting features.** With 19 positives, the number of features matters more than
   which clever ones you pick, and a single global ranking is meaningless at this sample
   size. So selection is exposed as a transformer that runs inside the fold, plus a
   stability report that shows how often each feature survives across folds.

Nothing here imports ``preprocessing`` (which imports this module), so there is one
direction of dependency and no cycle. Everything is a function of ``(X, y)`` passed in
by the caller.
"""

from __future__ import annotations

from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import mutual_info_classif


# --------------------------------------------------------------------------------------
# Feature creation
# --------------------------------------------------------------------------------------


class PriorHistoryFeatures(TransformerMixin, BaseEstimator):
    """Derived features for the prior-clicks numeric block.

    The EDA showed the effect of ``previous_clicks_on_similar_ads`` is a step, not a
    line: 15.8% click rate at or below 0.5, zero clicks above 2. A linear model cannot
    represent a step, so the flag is what lets logistic regression see the structure the
    trees find. ``log1p`` only helps the linear model cope with the right skew.

    ``low_history_cut`` came from looking at the whole dataset during EDA, so it carries
    a little optimism. It is a fixed constant rather than a fitted parameter, but the
    cut should be re-confirmed inside cross-validation.
    """

    def __init__(
        self,
        column: str = "previous_clicks_on_similar_ads",
        low_history_cut: float = 0.5,
        add_log: bool = True,
    ):
        self.column = column
        self.low_history_cut = low_history_cut
        self.add_log = add_log

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(
            list(X.columns)
            if isinstance(X, pd.DataFrame)
            else [f"x{i}" for i in range(np.asarray(X).shape[1])]
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)

        if self.column not in frame.columns:
            raise ValueError(f"column {self.column!r} not present")

        if self.add_log:
            frame[f"{self.column}__log1p"] = np.log1p(frame[self.column].clip(lower=0))
        frame[f"{self.column}__low_history"] = (
            frame[self.column] <= self.low_history_cut
        ).astype(float)
        return frame

    def get_feature_names_out(self, input_features=None):
        base = (
            list(input_features)
            if input_features is not None
            else list(self.feature_names_in_)
        )
        extra = [f"{self.column}__low_history"]
        if self.add_log:
            extra.insert(0, f"{self.column}__log1p")
        return np.asarray(base + extra, dtype=object)


class AgeBandFeatures(TransformerMixin, BaseEstimator):
    """One-hot age bands.

    ``user_age`` was flat in the EDA (p = 0.63), which is a reason to check whether the
    relationship is non-linear rather than a reason to drop the column. Banding gives
    the model a stepwise age effect without assuming linearity, at the cost of four
    extra columns — a real cost at 19 positives, so this is evaluated rather than
    assumed.
    """

    def __init__(
        self,
        column: str = "user_age",
        bins: tuple = (0, 25, 35, 45, 55, 120),
        labels: tuple = ("<25", "25-34", "35-44", "45-54", "55+"),
    ):
        self.column = column
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        if self.column not in getattr(X, "columns", []):
            raise ValueError(f"column {self.column!r} not present")
        self.categories_ = list(self.labels)
        return self

    def transform(self, X):
        binned = pd.cut(
            X[self.column], bins=list(self.bins), labels=list(self.labels), include_lowest=True
        )
        dummies = pd.get_dummies(binned, prefix=self.column, dtype=float)
        expected = [f"{self.column}_{label}" for label in self.categories_]
        return dummies.reindex(columns=expected, fill_value=0.0).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{self.column}_{l}" for l in self.categories_], dtype=object)


class PairwiseInteractionFeatures(TransformerMixin, BaseEstimator):
    """One-hot cross of two categorical columns.

    The EDA hinted that any remaining signal would come from an interaction around
    ``ad_category`` rather than from the demographic columns on their own. This turns
    that hypothesis into something testable instead of something assumed.

    Unseen combinations at transform time encode as all zeros, for the same reason
    ``handle_unknown="ignore"`` does upstream: a combination the model has never seen is
    evidence against the known ones, not a reason to crash.
    """

    def __init__(self, left: str, right: str, separator: str = "|"):
        self.left = left
        self.right = right
        self.separator = separator

    def _combine(self, X: pd.DataFrame) -> pd.Series:
        return (
            X[self.left].astype(str) + self.separator + X[self.right].astype(str)
        )

    def fit(self, X, y=None):
        for col in (self.left, self.right):
            if col not in getattr(X, "columns", []):
                raise ValueError(f"column {col!r} not present")
        self.levels_ = sorted(self._combine(X).unique())
        self.feature_names_in_ = np.asarray([self.left, self.right], dtype=object)
        return self

    def transform(self, X):
        combined = pd.Categorical(self._combine(X), categories=self.levels_)
        dummies = pd.get_dummies(combined, dtype=float)
        # get_dummies already carries the level as the column label, so rename before
        # reindexing — reindexing against the prefixed names without this step silently
        # produced an all-zero block.
        dummies.columns = [
            f"{self.left}__x__{self.right}={column}" for column in dummies.columns
        ]
        expected = [f"{self.left}__x__{self.right}={level}" for level in self.levels_]
        return dummies.reindex(columns=expected, fill_value=0.0).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [f"{self.left}__x__{self.right}={level}" for level in self.levels_], dtype=object
        )


# --------------------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------------------


class FeatureSelector(TransformerMixin, BaseEstimator):
    """In-pipeline feature selection.

    Deliberately a transformer rather than a preprocessing step you run once: selection
    that sees the test fold is the same leakage as scaling on the full frame, and it is
    the more common version of it in tabular work.

    ``strategy="mutual_info"`` keeps the top ``k`` by mutual information. ``strategy="l1"``
    keeps features with a non-zero L1 logistic coefficient, which is more stable at small
    sample sizes because it respects correlation between the one-hot columns. At 19
    positives neither is reliable — that is what ``selection_frequency`` is for.
    """

    def __init__(self, strategy: str = "l1", k: int = 12, random_state: int = 42):
        self.strategy = strategy
        self.k = k
        self.random_state = random_state

    def _make_selector(self):
        if self.strategy == "mutual_info":
            def score(X, y):
                return mutual_info_classif(X, y, random_state=self.random_state)

            return SelectKBest(score_func=score, k=min(self.k, np.inf))
        if self.strategy == "l1":
            estimator = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            )
            return SelectFromModel(estimator, threshold="median")
        raise ValueError(f"unknown strategy: {self.strategy!r}")

    def fit(self, X, y):
        self.selector_ = self._make_selector()
        self.selector_.fit(X, y)
        matrix = np.asarray(X)
        self.feature_names_in_ = np.asarray(
            list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(matrix.shape[1])]
        )
        self.support_ = self.selector_.get_support()
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_feature_names_out(self, input_features=None):
        names = (
            np.asarray(input_features, dtype=object)
            if input_features is not None
            else self.feature_names_in_
        )
        return names[self.support_]


def _selected_names(pipeline, selector_step: str = "select", preprocess_step: str = "preprocess"):
    """Names for the columns a fitted selector kept.

    ``ColumnTransformer`` hands the selector a numpy array, so the selector's own
    ``feature_names_in_`` is ``x0..xN`` and useless in a report. The preprocessor sitting
    in front of it knows the real names, so ask it and apply the support mask. Falls back
    to the selector's names if lengths don't line up.
    """
    selector = pipeline.named_steps[selector_step]
    support = selector.support_

    try:
        names = np.asarray(
            pipeline.named_steps[preprocess_step].get_feature_names_out(), dtype=object
        )
        if names.shape[0] == support.shape[0]:
            return names[support]
    except Exception:  # pragma: no cover - defensive
        pass
    return np.asarray(selector.get_feature_names_out(), dtype=object)


# --------------------------------------------------------------------------------------
# Selection diagnostics
# --------------------------------------------------------------------------------------


def mutual_information_table(X: pd.DataFrame, y, top_n: int = 15) -> pd.DataFrame:
    """Global MI ranking, for reference only.

    Included because the EDA used it, and because the honest way to present a global
    ranking is next to the fold-level stability table that shows it shifting.
    """
    scores = mutual_info_classif(X, y, random_state=42)
    out = (
        pd.DataFrame({"feature": X.columns, "mutual_information": scores})
        .sort_values("mutual_information", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    out["mutual_information"] = out["mutual_information"].round(5)
    return out


def selection_frequency(
    pipeline_factory,
    X: pd.DataFrame,
    y,
    *,
    selector_step: str = "select",
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """How often each feature is kept, across CV folds.

    A feature that survives every fold is a candidate for "answered". A feature that
    survives half of them is a coin flip, and at 19 positives most of this table will be
    coin flips. That is the result — not a problem with the method.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    n_fits = n_splits * n_repeats
    kept = Counter()

    for train_idx, _ in cv.split(X, y):
        pipeline = pipeline_factory()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        names = _selected_names(pipeline, selector_step)
        kept.update(set(names))

    rows = [
        {
            "feature": name,
            "folds_selected": count,
            "share_of_fits": count / n_fits,
        }
        for name, count in kept.items()
    ]
    table = pd.DataFrame(rows).sort_values(
        ["folds_selected", "feature"], ascending=[False, True]
    )
    table["n_fits"] = n_fits
    return table.reset_index(drop=True)


def evaluate_pipelines(
    X: pd.DataFrame,
    y,
    factories: dict,
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    flag_rate: float = 0.085,
    random_state: int = 42,
) -> pd.DataFrame:
    """Repeated stratified CV for a set of named pipeline factories.

    Reports mean ± std of average precision (PR-AUC) and of recall at a fixed flag rate,
    where the threshold comes from the training fold each time. Accuracy is not reported,
    for the reason established in the EDA: predicting none for everything scores 98%.

    Treat the std as the headline. With ~4 positives per fold, a mean difference smaller
    than the standard deviation is not a difference.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    results = []

    for name, factory in factories.items():
        ap_scores, recalls, n_features = [], [], []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline = factory()
            pipeline.fit(X_train, y_train)

            proba_train = pipeline.predict_proba(X_train)[:, 1]
            proba_test = pipeline.predict_proba(X_test)[:, 1]

            threshold = float(np.quantile(proba_train, 1 - flag_rate))
            predicted = proba_test >= threshold

            positives = int((y_test == 1).sum())
            tp = int((predicted & (y_test == 1)).sum())
            recalls.append(tp / max(positives, 1))

            if positives > 0 and 0 < len(np.unique(y_test)):
                ap_scores.append(average_precision_score(y_test, proba_test))
            else:
                ap_scores.append(np.nan)

            try:
                if "select" in pipeline.named_steps:
                    n_features.append(int(pipeline.named_steps["select"].support_.sum()))
                else:
                    n_features.append(
                        int(pipeline.named_steps["preprocess"].get_feature_names_out().shape[0])
                    )
            except Exception:  # pragma: no cover - defensive
                n_features.append(np.nan)

        results.append(
            {
                "feature_set": name,
                "n_features": int(np.nanmean(n_features)) if n_features else np.nan,
                "pr_auc_mean": round(float(np.nanmean(ap_scores)), 4),
                "pr_auc_std": round(float(np.nanstd(ap_scores)), 4),
                "recall_mean": round(float(np.mean(recalls)), 3),
                "recall_std": round(float(np.std(recalls)), 3),
            }
        )

    return pd.DataFrame(results).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


__all__ = [
    "PriorHistoryFeatures",
    "AgeBandFeatures",
    "PairwiseInteractionFeatures",
    "FeatureSelector",
    "mutual_information_table",
    "selection_frequency",
    "evaluate_pipelines",
]
