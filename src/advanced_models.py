"""Advanced models and none-detection for the ads "None" classification problem.

Two jobs, both relative to the baselines in ``models.py``:

1. **Two architectures that are not trees and not linear**: an RBF support vector machine
   (margin-based) and a small multilayer perceptron (neural). Both go through the same
   preprocessing pipeline, so any difference is the model and not the plumbing.
2. **None detection done properly.** ``models.none_score`` used 1 − max probability as a
   convenience. Here there are four scores — max probability, decision margin, energy,
   Mahalanobis distance and k-NN distance — and the comparison matters, because in a
   binary problem the first three are monotone transforms of each other and therefore
   identical in AUROC. Only the distance-based scores use feature geometry rather than
   the classifier head, so only they have a chance of behaving differently.

Everything is fitted on training folds only; the detector scores are fitted on the same
preprocessed training matrix the classifier saw.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42


# --------------------------------------------------------------------------------------
# None scores
# --------------------------------------------------------------------------------------


def energy_from_margin(margin) -> np.ndarray:
    """Negative free energy for a two-logit classifier.

    With logits ``[0, f]`` the energy is ``-log(1 + exp(f))``. Higher means more
    none-like. Monotone in ``-f``, which is exactly the point: for a binary classifier
    energy and the raw margin produce identical rankings.
    """
    margin = np.asarray(margin, dtype=float)
    return -np.logaddexp(0.0, margin)


class NoneDetector(BaseEstimator):
    """Fit a none-of-the-above score on training features, then score new rows.

    ``method="mahalanobis"`` uses a pooled Gaussian fitted to the training feature matrix;
    ``method="knn"`` uses the distance to the k-th nearest training row. Both operate on
    the preprocessed design matrix, so they see the same representation as the classifier.
    """

    def __init__(self, method: str = "mahalanobis", k: int = 5):
        self.method = method
        self.k = k

    def fit(self, X, y=None):
        matrix = np.asarray(X, dtype=float)
        self.n_features_in_ = matrix.shape[1]

        if self.method == "mahalanobis":
            self.mean_ = matrix.mean(axis=0)
            covariance = np.cov(matrix, rowvar=False)
            # Shrink towards the diagonal: 24 features and a few hundred rows leaves the
            # sample covariance close to singular, and an inverted near-singular matrix
            # produces distances dominated by numerical noise.
            covariance = (1 - 0.1) * covariance + 0.1 * np.diag(np.diag(covariance))
            self.precision_ = np.linalg.pinv(covariance)
        elif self.method == "knn":
            self.reference_ = matrix
        else:
            raise ValueError(f"unknown method: {self.method!r}")
        return self

    def score(self, X) -> np.ndarray:
        """Higher means more none-like."""
        matrix = np.asarray(X, dtype=float)
        if self.method == "mahalanobis":
            delta = matrix - self.mean_
            return np.sqrt(np.einsum("ij,jk,ik->i", delta, self.precision_, delta))
        if self.method == "knn":
            distances = np.linalg.norm(
                matrix[:, None, :] - self.reference_[None, :, :], axis=2
            )
            k = min(self.k, distances.shape[1])
            return np.sort(distances, axis=1)[:, k - 1]
        raise ValueError(f"unknown method: {self.method!r}")


def transformation_none_score(model, X, kind: str = "maxprob") -> np.ndarray:
    """None score from the classifier head itself; higher means more none-like."""
    if kind == "maxprob":
        return 1.0 - np.asarray(model.predict_proba(X)).max(axis=1)
    if kind == "margin":
        return -np.asarray(model.decision_function(X), dtype=float)
    if kind == "energy":
        return energy_from_margin(model.decision_function(X))
    raise ValueError(f"unknown kind: {kind!r}")


# --------------------------------------------------------------------------------------
# Advanced architectures
# --------------------------------------------------------------------------------------


def svm_pipeline(preprocessor=None, *, C: float = 1.0, gamma: str = "scale") -> Pipeline:
    """RBF SVM. Margin-based, so it assumes a different geometry to trees or logistic.

    ``probability=True`` costs an internal Platt calibration per fit; that is the only way
    to get comparable probabilities out of an SVM, and the Brier score later shows the
    cost of doing it on 15 positives per fold.
    """
    return Pipeline(
        [
            ("preprocess", preprocessor if preprocessor is not None else build_preprocessor()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def mlp_pipeline(
    preprocessor=None,
    *,
    hidden_layer_sizes=(32, 16),
    alpha: float = 1e-3,
    max_iter: int = 800,
) -> Pipeline:
    """Small MLP. Included because "advanced" usually implies one, not because it suits
    the data: 19 positives is far below what a neural network needs, and the literature
    on tabular data says tree ensembles usually win regardless.
    """
    return Pipeline(
        [
            ("preprocess", preprocessor if preprocessor is not None else build_preprocessor()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=alpha,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=20,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def knn_pipeline(preprocessor=None, *, n_neighbors: int = 15) -> Pipeline:
    """k-NN classifier, kept as a distance-based reference point for the detectors."""
    return Pipeline(
        [
            ("preprocess", preprocessor if preprocessor is not None else build_preprocessor()),
            ("model", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")),
        ]
    )


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------


def paired_comparison(folds: pd.DataFrame, reference: str, metric: str = "pr_auc") -> pd.DataFrame:
    """Per-fold deltas against a reference model.

    Same folds for every model, so the comparison is paired and the spread of the delta
    is more informative than the spread of either score. ``wins`` counts folds where the
    model beat the reference — with 50 folds, 30 wins is a coin flip and 45 is not.
    """
    unique = folds.drop_duplicates(["model", "fold"])
    ref = unique[unique["model"] == reference].set_index("fold")[metric]

    rows = []
    for name in unique["model"].unique():
        if name == reference:
            continue
        scores = unique[unique["model"] == name].set_index("fold")[metric]
        delta = (scores - ref).dropna()
        if delta.empty:
            continue
        rows.append(
            {
                "model": name,
                "mean_delta": round(float(delta.mean()), 4),
                "std_delta": round(float(delta.std()), 4),
                "wins": int((delta > 0).sum()),
                "folds": int(len(delta)),
                "win_rate": round(float((delta > 0).mean()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_delta", ascending=False).reset_index(drop=True)


def none_detector_comparison(
    known_train_X,
    known_train_y,
    known_test_X,
    unknown_X,
    *,
    classifier_factory=svm_pipeline,
) -> pd.DataFrame:
    """AUROC / AUPR for each none score on the same held-out unknown set.

    The classifier is fitted on known classes only. Every score is then computed on the
    held-out known rows (negatives) and the unknown rows (positives) — the detection
    question, not the click question.
    """
    model = classifier_factory()
    model.fit(known_train_X, known_train_y)

    preprocessor = model.named_steps["preprocess"]
    Z_known = preprocessor.transform(known_test_X)
    Z_unknown = preprocessor.transform(unknown_X)

    labels = np.concatenate([np.zeros(len(Z_known)), np.ones(len(Z_unknown))])

    candidates = {
        "max probability": transformation_none_score(model, known_test_X, "maxprob"),
        "decision margin": transformation_none_score(model, known_test_X, "margin"),
        "energy": transformation_none_score(model, known_test_X, "energy"),
    }
    unknown_scores = {
        "max probability": transformation_none_score(model, unknown_X, "maxprob"),
        "decision margin": transformation_none_score(model, unknown_X, "margin"),
        "energy": transformation_none_score(model, unknown_X, "energy"),
    }

    for method in ("mahalanobis", "knn"):
        detector = NoneDetector(method=method).fit(Z_known)
        candidates[method] = detector.score(Z_known)
        unknown_scores[method] = detector.score(Z_unknown)

    rows = []
    for name, known_scores in candidates.items():
        combined = np.concatenate([known_scores, unknown_scores[name]])
        threshold = float(np.quantile(known_scores, 0.95))
        rows.append(
            {
                "score": name,
                "auroc": round(float(roc_auc_score(labels, combined)), 3),
                "aupr": round(float(average_precision_score(labels, combined)), 3),
                "detection_at_5fpr": round(float((unknown_scores[name] >= threshold).mean()), 3),
                "known_mean": round(float(np.mean(known_scores)), 3),
                "unknown_mean": round(float(np.mean(unknown_scores[name])), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


__all__ = [
    "energy_from_margin",
    "NoneDetector",
    "transformation_none_score",
    "svm_pipeline",
    "mlp_pipeline",
    "knn_pipeline",
    "paired_comparison",
    "none_detector_comparison",
]
