"""Sprint 3 smoke test and validation for the preprocessing pipeline.

Run from anywhere:

    python main.py

It builds the pipeline, fits it on a training split, checks the properties the None
problem depends on (no leakage of full-data statistics, no NaN in the output, unseen
categories handled, deterministic output), and finishes with a small logistic-regression
baseline so the pipeline is exercised end to end. This is a validation harness, not the
modelling sprint — nothing here is meant to be reported as a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_model,
    build_preprocessor,
    filter_eligible_rows,
    load_data,
    make_open_set_split,
    make_train_test_split,
    positive_rate,
    transform_frame,
)

RANDOM_STATE = 42
TARGET_FLAG_RATE = 0.085  # the single-rule baseline from the EDA


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(label)


def main() -> None:
    rule("1. Load, validate, filter")

    df = load_data()
    print(f"loaded {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"positive rate before filtering: {positive_rate(df[TARGET]):.2%}")

    df, report = filter_eligible_rows(df)
    print(f"filter report: {report}")
    check("under-18 rows removed", report["under_min_age"] > 0, f"{report['under_min_age']} rows")
    check("no rows lost to duplicates", report["duplicate_rows"] == 0)
    check(
        "positive count reconciles",
        report["positives_out"] == report["positives_in"] - report["positives_dropped"],
    )
    print(
        f"cost of the age filter: {report['positives_dropped']} of "
        f"{report['positives_in']} positives dropped "
        f"({report['positives_dropped'] / report['positives_in']:.1%} of the minority class)"
    )
    check("no nulls in kept frame", not df.isna().any().any())

    rule("2. Stratified split")

    X_train, X_test, y_train, y_test = make_train_test_split(df, test_size=0.2)
    print(f"train: {X_train.shape}, positives {int(y_train.sum())} ({positive_rate(y_train):.2%})")
    print(f"test:  {X_test.shape}, positives {int(y_test.sum())} ({positive_rate(y_test):.2%})")
    check("no row overlap between splits", len(set(X_train.index) & set(X_test.index)) == 0)
    check("test split contains positives", int(y_test.sum()) > 0)

    rule("3. Fit the preprocessor on the training split only")

    pre = build_preprocessor()
    train_matrix = pre.fit_transform(X_train)
    test_matrix = pre.transform(X_test)
    names = list(pre.get_feature_names_out())

    print(f"transformed train: {train_matrix.shape}, test: {test_matrix.shape}")
    print(f"output features ({len(names)}): {names}")
    check("no NaN after transform", not np.isnan(train_matrix).any(), "train")
    check("no NaN after transform", not np.isnan(test_matrix).any(), "test")
    check(
        "feature count is stable across splits",
        train_matrix.shape[1] == test_matrix.shape[1],
    )

    expected_numeric = len(NUMERIC_FEATURES) + 2  # log1p + low_history
    expected = expected_numeric + sum(X_train[c].nunique() for c in CATEGORICAL_FEATURES)
    check("feature count matches the schema", len(names) == expected, f"{len(names)} == {expected}")

    low_history_cols = [c for c in names if c.endswith("__low_history")]
    check("low-history flag exists", len(low_history_cols) == 1)

    flag_col = names.index(low_history_cols[0])
    scaled_flag = train_matrix[:, flag_col]
    raw_flag = (X_train["previous_clicks_on_similar_ads"] <= 0.5).to_numpy()
    check(
        "low-history flag survived preprocessing",
        np.unique(scaled_flag).size == 2,
        f"unique scaled values: {np.unique(scaled_flag).round(3)}",
    )
    check(
        "low-history flag is aligned with the raw rule",
        scaled_flag[raw_flag].mean() > scaled_flag[~raw_flag].mean(),
        f"mean when true {scaled_flag[raw_flag].mean():.2f} vs false "
        f"{scaled_flag[~raw_flag].mean():.2f}",
    )
    print(f"low-history rows in train: {raw_flag.sum()} of {len(raw_flag)} "
          f"({raw_flag.mean():.1%})")

    rule("4. Leakage checks: statistics come from train, not the full frame")

    numeric_pipe = pre.named_transformers_["numeric"]
    engineer = numeric_pipe.named_steps["engineer"]
    clipper = numeric_pipe.named_steps["clip"]
    scaler = numeric_pipe.named_steps["scale"]

    engineered_train = engineer.transform(X_train)
    engineered_full = engineer.transform(df[list(FEATURES)])
    numeric_names = list(engineer.get_feature_names_out())

    for i, col in enumerate(numeric_names):
        train_values = engineered_train[col].to_numpy()
        full_values = engineered_full[col].to_numpy()
        train_q1, train_q3 = np.nanpercentile(train_values, [25, 75])
        train_iqr = train_q3 - train_q1
        train_fence = train_q3 + 1.5 * train_iqr
        full_q1, full_q3 = np.nanpercentile(full_values, [25, 75])
        full_fence = full_q3 + 1.5 * (full_q3 - full_q1)
        print(
            f"{col}: train fence {train_fence:.3f} vs full-data fence {full_fence:.3f} "
            f"| stored {clipper.upper_[i]:.3f}"
        )
        if train_iqr == 0:
            check(f"{col} has no spread so is left unclipped", np.isinf(clipper.upper_[i]))
        else:
            check(f"{col} fence fitted on train", np.isclose(clipper.upper_[i], train_fence))

    for i, col in enumerate(numeric_names):
        train_values = engineered_train[col].to_numpy()
        clipped_train = np.clip(train_values, clipper.lower_[i], clipper.upper_[i])
        print(f"{col}: stored mean {scaler.mean_[i]:.3f} | clipped train mean "
              f"{np.nanmean(clipped_train):.3f} | full mean {np.nanmean(engineered_full[col]):.3f}")
        check(
            f"{col} scaler fitted on clipped train",
            np.isclose(scaler.mean_[i], np.nanmean(clipped_train)),
        )

    rule("5. Unseen categories fall back to an all-zero block")

    probe = X_test.head(3).copy()
    probe.loc[probe.index[0], "ad_category"] = "Podcast"  # never appears in training
    transformed = transform_frame(pre, probe)
    category_cols = [c for c in transformed.columns if "ad_category" in c]
    print(f"ad_category block for the injected row: {transformed.loc[probe.index[0], category_cols].to_dict()}")
    check(
        "unseen category encoded as all zeros",
        np.allclose(transformed.loc[probe.index[0], category_cols].to_numpy(), 0.0),
    )
    check("row still transforms", len(transformed) == len(probe))

    rule("6. Determinism")

    again = build_preprocessor().fit_transform(X_train)
    check("two fits give identical output", np.array_equal(train_matrix, again))

    rule("7. End-to-end: logistic regression through the pipeline")

    model = build_model(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    )
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    proba_train = model.predict_proba(X_train)[:, 1]

    # Threshold chosen on training scores at the EDA flag rate, never on the test set.
    threshold = float(np.quantile(proba_train, 1 - TARGET_FLAG_RATE))
    pred = (proba_test >= threshold).astype(int)

    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    flag_rate = (pred == 1).mean()

    print(f"threshold {threshold:.3f} (chosen on train at {TARGET_FLAG_RATE:.1%} flag rate)")
    print(f"test: TP {tp}, FP {fp}, FN {fn} | recall {recall:.2f}, precision {precision:.2f}, "
          f"flag rate {flag_rate:.1%}")
    print("EDA single-rule baseline: recall 0.80, precision 0.16 at an 8.5% flag rate")
    print("(test positives are under 10 rows, so treat the difference as noise)")

    check("pipeline predicts probabilities", proba_test.min() >= 0 and proba_test.max() <= 1)

    rule("8. Open-set smoke test (None behaviour)")

    X_tr, X_te, y_tr, y_te, unknown_df = make_open_set_split(df, "Travel")
    os_model = build_model(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    )
    os_model.fit(X_tr, y_tr)
    known_scores = os_model.predict_proba(X_te)[:, 1]
    unknown_scores = os_model.predict_proba(unknown_df[list(FEATURES)])[:, 1]

    print(f"held-out category 'Travel': {len(unknown_df)} rows, {int(unknown_df[TARGET].sum())} positives")
    print(f"mean predicted click probability — known test {known_scores.mean():.3f}, "
          f"unknown rows {unknown_scores.mean():.3f}")
    print("If the unknown rows score like known rows, the model cannot tell 'none of my "
          "classes' from a known class. That is the gap Sprint 1 describes and Sprint 4 "
          "has to measure.")
    check("open-set split returns rows from the held-out category", len(unknown_df) > 0)
    check("held-out category absent from training", "Travel" not in set(X_tr["ad_category"]))

    rule("Summary")

    print(f"rows used: {len(df)} | features in: {len(FEATURES)} | features out: {len(names)}")
    print(f"base positive rate: {positive_rate(df[TARGET]):.2%}")
    print(f"pipeline: {model}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
