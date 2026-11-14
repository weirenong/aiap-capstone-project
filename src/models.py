"""Baseline models and evaluation for the ads "None" classification problem.

Two things live here:

1. **Baselines.** A floor (always predict none), the EDA's single-threshold rule, and
   three classifiers through the Sprint 3 pipeline. Every later model has to beat the
   rule, not just the floor.
2. **Metrics that fit the problem.** Accuracy is useless at a 2% positive rate, so the
   reporting is built around PR-AUC and recall at a fixed flag rate, with the operating
   threshold taken from the training fold each time.

There is also an **open-set report**. The click target only exercises the "none as
background" reading of the problem; the other reading — can the model tell that a class
it never saw is none-of-the-above — needs its own evaluation, and Sprint 3 showed the
answer was no. That gets measured here rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

NONE_LABEL = 0
POSITIVE_LABEL = 1

DEFAULT_FLAG_RATES = (0.05, 0.085, 0.10)
DEFAULT_RULE_CUT = 0.5
DEFAULT_RULE_COLUMN = "previous_clicks_on_similar_ads"


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------


class AlwaysNoneClassifier(ClassifierMixin, BaseEstimator):
    """Predicts the none class for every row.

    This is not a strawman that can be skipped — on a 2% positive rate it is 98%
    accurate. Reporting it next to the real models is what makes the accuracy point
    concrete, and it is the model any "none-first" system degrades to if nothing is
    tuned.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([NONE_LABEL, POSITIVE_LABEL])
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.zeros(np.asarray(X).shape[0], dtype=int)

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        return np.column_stack([np.ones(n), np.zeros(n)])


class PriorHistoryRuleClassifier(ClassifierMixin, BaseEstimator):
    """The EDA's single-threshold rule: low prior clicks means likely to click.

    ``decision_function`` is ``cut - previous_clicks``, so larger means "more likely to
    click" and the rule is recovered exactly by thresholding at 0. That gives a proper
    ranking for PR-AUC without inventing calibrated probabilities — the rule has no
    probability model behind it and shouldn't pretend otherwise.
    """

    def __init__(self, column: str = DEFAULT_RULE_COLUMN, cut: float = DEFAULT_RULE_CUT):
        self.column = column
        self.cut = cut

    def fit(self, X, y=None):
        if self.column not in getattr(X, "columns", []):
            raise ValueError(f"column {self.column!r} not present")
        self.classes_ = np.array([NONE_LABEL, POSITIVE_LABEL])
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def decision_function(self, X):
        return (self.cut - X[self.column]).to_numpy(dtype=float)

    def predict(self, X):
        return (X[self.column] <= self.cut).to_numpy(dtype=int)

    def predict_proba(self, X):
        # Monotone squashing of the decision, for callers that require probabilities.
        # Not calibrated, and not used for any reported metric.
        margin = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-margin))
        return np.column_stack([1 - p, p])


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def scores_from_model(model, X) -> np.ndarray:
    """Probability of the positive class, or a decision score if probabilities are absent."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).astype(float)
    raise TypeError(f"{type(model).__name__} exposes neither predict_proba nor decision_function")


def threshold_at_flag_rate(scores, flag_rate: float) -> float:
    """Highest threshold that flags at most ``flag_rate`` of the rows.

    Must be called with training-fold scores. Calling it on the evaluation rows is the
    usual way a "recall at 5% flag rate" number turns into fiction.

    Ties matter here. A constant score — the always-none floor — has a 95th percentile
    equal to every other value, so a plain ``quantile`` threshold flags every row,
    "achieves" recall 1.0 and makes the floor look like a detector. Walking the distinct
    values and stopping at the first one inside the budget makes the floor flag nothing,
    which is the truth.
    """
    scores = np.asarray(scores, dtype=float)
    if not 0 < flag_rate < 1:
        raise ValueError("flag_rate must be between 0 and 1")

    # Ascending: the share flagged falls as the threshold rises, so the first value that
    # fits the budget is the lowest threshold that does — i.e. the most recall available.
    for value in np.unique(scores):
        if (scores >= value).mean() <= flag_rate + 1e-12:
            return float(value)
    return float(np.nextafter(scores.max(), np.inf))


def confusion_at(y_true, scores, threshold: float) -> dict:
    """Counts and derived rates at one operating point."""
    y_true = np.asarray(y_true)
    predicted = (np.asarray(scores, dtype=float) >= threshold).astype(int)

    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flag_rate": float(predicted.mean()),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }


def ranking_metrics(y_true, scores, probabilities=None) -> dict:
    """Threshold-free metrics. Skipped rather than faked when a class is missing."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    out = {"n": int(len(y_true)), "positives": int((y_true == 1).sum())}
    if out["positives"] == 0 or out["positives"] == out["n"]:
        out.update({"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan})
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, scores))
    out["pr_auc"] = float(average_precision_score(y_true, scores))
    out["brier"] = (
        float(brier_score_loss(y_true, probabilities)) if probabilities is not None else np.nan
    )
    return out


def evaluate_at_flag_rates(
    y_true,
    scores,
    *,
    flag_rates=DEFAULT_FLAG_RATES,
    threshold_scores=None,
    probabilities=None,
) -> pd.DataFrame:
    """One row per operating point, plus the threshold-free metrics repeated for context.

    ``threshold_scores`` are the scores used to *choose* each threshold (normally the
    training fold). Defaults to ``scores``, which is only correct for a final held-out
    report, not for cross-validation.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    reference = np.asarray(threshold_scores, dtype=float) if threshold_scores is not None else scores

    ranking = ranking_metrics(y_true, scores, probabilities=probabilities)
    rows = []

    for rate in flag_rates:
        threshold = threshold_at_flag_rate(reference, rate)
        row = {"flag_rate_target": rate, "threshold": threshold}
        row.update(confusion_at(y_true, scores, threshold))
        row.update({k: ranking[k] for k in ("roc_auc", "pr_auc")})
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Cross-validated comparison
# --------------------------------------------------------------------------------------


def cross_validate_models(
    X,
    y,
    factories: dict,
    *,
    flag_rates=DEFAULT_FLAG_RATES,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated stratified CV returning per-fold rows and a mean/std summary.

    Thresholds come from the training fold, metrics from the test fold, so nothing here
    can see the row it is scored on. Returns ``(folds, summary)`` — keep the folds, since
    a summary without the spread hides how little 19 positives can support.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    fold_rows = []

    for name, factory in factories.items():
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = factory()
            model.fit(X_train, y_train)

            train_scores = scores_from_model(model, X_train)
            test_scores = scores_from_model(model, X_test)

            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = np.asarray(model.predict_proba(X_test))[:, 1]

            ranking = ranking_metrics(y_test, test_scores, probabilities=probabilities)

            for rate in flag_rates:
                threshold = threshold_at_flag_rate(train_scores, rate)
                row = {
                    "model": name,
                    "fold": fold,
                    "flag_rate_target": rate,
                    "threshold": threshold,
                    **ranking,
                }
                row.update(confusion_at(y_test, test_scores, threshold))
                fold_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    agg = (
        folds.groupby(["model", "flag_rate_target"])
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            brier_mean=("brier", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            f1_mean=("f1", "mean"),
            achieved_flag_rate=("flag_rate", "mean"),
        )
        .round(4)
        .reset_index()
    )
    return folds, agg


# --------------------------------------------------------------------------------------
# Open-set ("none-of-the-above") evaluation
# --------------------------------------------------------------------------------------


def none_score(model, X) -> np.ndarray:
    """How much each row looks like none-of-the-above; higher means more none-like.

    Uses one minus the maximum probability when the model emits probabilities, and the
    negative maximum decision otherwise. Deliberately simple: Sprint 1's point is that a
    closed-set classifier plus a score is the baseline any dedicated none-head has to
    beat, so this is that baseline made explicit.
    """
    if hasattr(model, "predict_proba"):
        return 1.0 - np.asarray(model.predict_proba(X)).max(axis=1)
    decisions = np.asarray(model.decision_function(X)).astype(float)
    return -decisions


@dataclass
class OpenSetReport:
    """Result container so callers don't have to remember dict keys."""

    auroc: float
    aupr: float
    detection_rate_at_5fpr: float
    known_mean_none_score: float
    unknown_mean_none_score: float
    n_known: int
    n_unknown: int
    threshold_at_5fpr: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def open_set_report(
    known_train_X,
    known_train_y,
    known_test_X,
    unknown_X,
    factory,
) -> tuple[OpenSetReport, object]:
    """Fit on known classes only, then ask whether held-out classes look like none.

    The detection question is binary and separate from click prediction: every held-out
    row is a positive for "this is none-of-the-above", every known test row is a negative.
    That is why AUROC is defensible here even though the click task has 19 positives —
    the unknown set has its own, larger sample.

    Returns the report and the fitted model (the caller usually wants it for inspection).
    """
    model = factory()
    model.fit(known_train_X, known_train_y)

    known_scores = none_score(model, known_test_X)
    unknown_scores = none_score(model, unknown_X)

    combined = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate(
        [np.zeros(len(known_scores), dtype=int), np.ones(len(unknown_scores), dtype=int)]
    )

    auroc = float(roc_auc_score(labels, combined)) if len(np.unique(labels)) == 2 else np.nan
    aupr = float(average_precision_score(labels, combined)) if len(np.unique(labels)) == 2 else np.nan

    threshold = float(np.quantile(known_scores, 0.95))
    detection_rate = float((unknown_scores >= threshold).mean())

    report = OpenSetReport(
        auroc=auroc,
        aupr=aupr,
        detection_rate_at_5fpr=detection_rate,
        known_mean_none_score=float(known_scores.mean()),
        unknown_mean_none_score=float(unknown_scores.mean()),
        n_known=int(len(known_scores)),
        n_unknown=int(len(unknown_scores)),
        threshold_at_5fpr=threshold,
    )
    return report, model


def open_set_auc_by_category(
    known_train_X,
    known_train_y,
    known_test_X,
    unknown_frames: dict,
    factory,
) -> pd.DataFrame:
    """Run the open-set report once per held-out category.

    Semantic shift is the dominant factor in open-set difficulty (Vaze et al.), and the
    categories here differ in size and click rate, so a single held-out set is not enough
    to conclude anything. This is also the honest way to show how thin the evidence is.
    """
    rows = []
    for category, unknown_df in unknown_frames.items():
        report, _ = open_set_report(
            known_train_X,
            known_train_y,
            known_test_X,
            unknown_df[known_train_X.columns],
            factory,
        )
        rows.append(
            {
                "holdout_category": category,
                "n_unknown": report.n_unknown,
                "auroc": round(report.auroc, 3),
                "aupr": round(report.aupr, 3),
                "detection_at_5fpr": round(report.detection_rate_at_5fpr, 3),
                "unknown_mean_none_score": round(report.unknown_mean_none_score, 3),
                "known_mean_none_score": round(report.known_mean_none_score, 3),
            }
        )
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


__all__ = [
    "AlwaysNoneClassifier",
    "PriorHistoryRuleClassifier",
    "scores_from_model",
    "threshold_at_flag_rate",
    "confusion_at",
    "ranking_metrics",
    "evaluate_at_flag_rates",
    "cross_validate_models",
    "none_score",
    "OpenSetReport",
    "open_set_report",
    "open_set_auc_by_category",
]
