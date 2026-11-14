"""Cross-validation and hyperparameter tuning for the ads "None" classification problem.

The point of this module is to tune without lying about the result.

Tuning a model on the same folds you then report is the most common way a project
manufactures an improvement: the search picks whichever configuration happens to fit
that particular split, and the reported score includes the picking. With 19 positives,
that bias is large enough to invent a winner. So the estimate here is **nested**:

    outer split  -> held out, never touched during tuning
      inner splits on the outer-train part -> GridSearchCV picks parameters
      refit on the whole outer-train part
      score once on the outer-test part

The same outer folds also score the *default* configuration, so "tuning helped" is a
paired per-fold difference rather than a comparison of two unrelated numbers.

Two further diagnostics live here, because tuning alone cannot answer the question the
sprint asks:

- ``tuning_stability`` — which parameters win across outer folds. If the winners move
  from fold to fold, the search is fitting noise and the "best" params mean nothing.
- ``learning_curve_report`` — train vs cross-validated score as the training set grows,
  which separates overfitting (big gap, improving validation) from underfitting (both
  low) and from "the problem is small" (both flat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.feature_engineering import FeatureSelector
from src.preprocessing import build_preprocessor

RANDOM_STATE = 42
SCORING = "average_precision"


# --------------------------------------------------------------------------------------
# Search spaces
# --------------------------------------------------------------------------------------
# Deliberately small. A grid with hundreds of points on 19 positives does not search
# better, it searches until something fits the fold. These widths are also chosen so the
# regularisation axis is covered in both directions: an underfitting model and an
# overfitting model both appear as candidates.


def search_spaces(include_selector: bool = False) -> dict[str, dict]:
    """Parameter grids keyed by model name. Prefixes match the pipeline steps."""
    spaces = {
        "logistic regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
        },
        "random forest": {
            "model__max_depth": [3, 6, None],
            "model__min_samples_leaf": [1, 5],
        },
        "rbf svm": {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", 0.01],
        },
        "knn": {
            "model__n_neighbors": [5, 15, 35],
            "model__weights": ["uniform", "distance"],
        },
        "mlp": {
            "model__alpha": [1e-4, 1e-2],
            "model__hidden_layer_sizes": [(16,), (32, 16)],
        },
    }
    if include_selector:
        spaces["logistic + select"] = {
            "model__C": [0.1, 1.0],
            "select__k": [8, 16],
        }
    return spaces


def model_factories(include_selector: bool = False) -> dict[str, callable]:
    """Pipeline factories matching the search-space names.

    The preprocessor is rebuilt per factory so every fit starts clean; sharing one
    instance across fits would leak fitted state between folds.
    """

    def logistic(**params):
        """Logistic regression pipeline, parameterised so the search grid can pass C."""
        return Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        **params,
                    ),
                ),
            ]
        )

    factories = {
        "logistic regression": logistic,
        "random forest": lambda: Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "rbf svm": lambda: Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "knn": lambda: Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("model", KNeighborsClassifier()),
            ]
        ),
        "mlp": lambda: Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    MLPClassifier(
                        max_iter=800,
                        early_stopping=True,
                        n_iter_no_change=20,
                        validation_fraction=0.15,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if include_selector:
        factories["logistic + select"] = lambda: Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("select", FeatureSelector(strategy="mutual_info", k=12)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    return factories


# --------------------------------------------------------------------------------------
# Nested cross-validation
# --------------------------------------------------------------------------------------


def _outer_splits(y, n_splits: int, n_repeats: int, random_state: int):
    from sklearn.model_selection import RepeatedStratifiedKFold

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    return list(cv.split(np.zeros(len(y)), y))


def nested_cv(
    X,
    y,
    *,
    model_names: tuple[str, ...] | None = None,
    outer_splits: int = 5,
    outer_repeats: int = 2,
    inner_splits: int = 3,
    flag_rate: float = 0.085,
    reference_factory=None,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested CV with a default-vs-tuned comparison on identical outer folds.

    If ``reference_factory`` is given (the prior-history rule), it is scored on the same
    outer folds, so the model-vs-reference comparison is paired like the tuned-vs-default
    one. Comparing against a number from a different CV run would be a different and
    weaker claim.

    Returns ``(fold_rows, param_rows)``. ``fold_rows`` has one row per model per outer
    fold with the tuned and default scores side by side; ``param_rows`` records the
    parameters each outer fold selected, which is what ``tuning_stability`` consumes.
    """
    spaces = search_spaces()
    factories = model_factories()
    names = model_names or tuple(spaces)

    splits = _outer_splits(y, outer_splits, outer_repeats, random_state)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    fold_rows, param_rows = [], []

    for name in names:
        space = spaces[name]
        factory = factories[name]

        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            search = GridSearchCV(
                factory(),
                param_grid=space,
                scoring=SCORING,
                cv=inner_cv,
                refit=True,
                n_jobs=1,
            )
            search.fit(X_train, y_train)
            tuned_proba = search.predict_proba(X_test)[:, 1]

            default = factory()
            default.fit(X_train, y_train)
            default_proba = default.predict_proba(X_test)[:, 1]

            threshold = float(
                np.quantile(search.predict_proba(X_train)[:, 1], 1 - flag_rate)
            )

            pos = int((y_test == 1).sum())

            reference_ap = np.nan
            if reference_factory is not None:
                reference = reference_factory()
                reference.fit(X_train, y_train)
                reference_ap = (
                    average_precision_score(y_test, reference.predict_proba(X_test)[:, 1])
                    if pos
                    else np.nan
                )

            tuned_pred = tuned_proba >= threshold
            default_pred = default_proba >= threshold

            fold_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "n_test_positives": pos,
                    "tuned_pr_auc": average_precision_score(y_test, tuned_proba)
                    if pos
                    else np.nan,
                    "default_pr_auc": average_precision_score(y_test, default_proba)
                    if pos
                    else np.nan,
                    "tuned_roc_auc": roc_auc_score(y_test, tuned_proba) if pos else np.nan,
                    "tuned_recall": float((tuned_pred & (y_test == 1)).sum() / max(pos, 1)),
                    "default_recall": float((default_pred & (y_test == 1)).sum() / max(pos, 1)),
                    "inner_best_score": float(search.best_score_),
                    "reference_pr_auc": reference_ap,
                }
            )

            for param, value in search.best_params_.items():
                param_rows.append(
                    {"model": name, "fold": fold, "parameter": param, "value": str(value)}
                )

    return pd.DataFrame(fold_rows), pd.DataFrame(param_rows)


def summarize_nested(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of tuned vs default, plus the paired gain, per model."""
    rows = []
    for name, group in folds.groupby("model"):
        gain = group["tuned_pr_auc"] - group["default_pr_auc"]
        rows.append(
            {
                "model": name,
                "tuned_pr_auc": round(float(group["tuned_pr_auc"].mean()), 4),
                "tuned_std": round(float(group["tuned_pr_auc"].std()), 4),
                "default_pr_auc": round(float(group["default_pr_auc"].mean()), 4),
                "gain": round(float(gain.mean()), 4),
                "gain_std": round(float(gain.std()), 4),
                "folds_better": int((gain > 0).sum()),
                "folds": int(len(group)),
                "inner_best": round(float(group["inner_best_score"].mean()), 4),
                "outer_minus_inner": round(
                    float(group["tuned_pr_auc"].mean() - group["inner_best_score"].mean()), 4
                ),
                "reference_pr_auc": round(float(group["reference_pr_auc"].mean()), 4)
                if group["reference_pr_auc"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("tuned_pr_auc", ascending=False).reset_index(drop=True)


def tuning_stability(params: pd.DataFrame) -> pd.DataFrame:
    """How often each parameter value was selected by the inner search.

    A value chosen in 14/20 folds is a preference; a value chosen in 7/20 is a coin flip
    and should not be reported as "the tuned configuration".
    """
    rows = []
    for (model, parameter), group in params.groupby(["model", "parameter"]):
        counts = group["value"].value_counts()
        total = int(counts.sum())
        for value, count in counts.items():
            rows.append(
                {
                    "model": model,
                    "parameter": parameter,
                    "value": value,
                    "selected": int(count),
                    "folds": total,
                    "share": round(float(count / total), 3),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["model", "parameter", "selected"], ascending=[True, True, False]
    ).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Over- and underfitting
# --------------------------------------------------------------------------------------


def learning_curve_report(
    factory,
    X,
    y,
    *,
    train_sizes=(20, 40, 60, 80, 100),
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Train vs cross-validated PR-AUC as the training set grows.

    Reads three ways:

    - train high, validation much lower -> overfitting; the gap is the diagnosis and the
      fix is capacity or regularisation, not more tuning.
    - both low and flat -> underfitting; the model is the wrong shape for the data.
    - both roughly equal and flat -> the model has extracted what is there, and the
      ceiling is the data, not the estimator.

    ``train_sizes`` are absolute row counts rather than fractions, because at 19
    positives fractions hide how few positives each point contains.
    """
    sizes, train_scores, cv_scores = learning_curve(
        factory(),
        X,
        y,
        train_sizes=list(train_sizes),
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
        scoring=SCORING,
        shuffle=True,
        random_state=random_state,
        error_score=np.nan,
    )
    return pd.DataFrame(
        {
            "train_rows": sizes.astype(int),
            "train_pr_auc": np.nanmean(train_scores, axis=1).round(3),
            "train_std": np.nanstd(train_scores, axis=1).round(3),
            "cv_pr_auc": np.nanmean(cv_scores, axis=1).round(3),
            "cv_std": np.nanstd(cv_scores, axis=1).round(3),
            "gap": (np.nanmean(train_scores, axis=1) - np.nanmean(cv_scores, axis=1)).round(3),
        }
    )


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


def select_final_model(
    summary: pd.DataFrame,
    *,
    reference_pr_auc: float,
    reference_name: str = "prior-history rule",
    tie_tolerance: float = 0.02,
) -> dict:
    """Pick a final model, preferring the simplest thing that is not measurably worse.

    The rule is: a tuned model has to beat the reference by more than ``tie_tolerance``
    *and* by more than its own fold-to-fold std to be preferred. Otherwise the reference
    wins, because it has no fitted parameters to go stale and no tuning bias to leak.
    """
    best = summary.iloc[0]
    margin = float(best["tuned_pr_auc"]) - reference_pr_auc
    decisive = margin > tie_tolerance and margin > float(best["tuned_std"])

    if decisive:
        return {
            "selected": str(best["model"]),
            "reason": (
                f"beats {reference_name} by {margin:+.3f} PR-AUC, which is larger than "
                f"both the {tie_tolerance:.2f} tie tolerance and its own std "
                f"({best['tuned_std']:.3f})"
            ),
            "margin_over_reference": round(margin, 4),
            "decisive": True,
        }

    return {
        "selected": reference_name,
        "reason": (
            f"best tuned model ({best['model']}) scores {best['tuned_pr_auc']:.3f} PR-AUC "
            f"against the reference's {reference_pr_auc:.3f} (margin {margin:+.3f}), which "
            f"is inside the {tie_tolerance:.2f} tie tolerance or below its own std "
            f"({best['tuned_std']:.3f}); the reference has no fitted parameters to go "
            f"stale, so it wins the tie"
        ),
        "margin_over_reference": round(margin, 4),
        "decisive": False,
    }


__all__ = [
    "search_spaces",
    "model_factories",
    "nested_cv",
    "summarize_nested",
    "tuning_stability",
    "learning_curve_report",
    "select_final_model",
]
