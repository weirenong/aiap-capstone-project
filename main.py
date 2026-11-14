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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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

    rule("9. Feature engineering: what actually earns its place")

    from src.feature_engineering import (
        AgeBandFeatures,
        FeatureSelector,
        PairwiseInteractionFeatures,
        evaluate_pipelines,
        selection_frequency,
    )

    def estimator():
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )

    def assemble(pre, selector=None):
        steps = [("preprocess", pre)]
        if selector is not None:
            steps.append(("select", selector))
        steps.append(("model", estimator()))
        return Pipeline(steps)

    def extend(pre, blocks):
        return ColumnTransformer(
            transformers=list(pre.transformers) + blocks,
            remainder="drop",
            verbose_feature_names_out=True,
        )

    history_pre = build_preprocessor()
    age_pre = extend(history_pre, [("age_bands", AgeBandFeatures(), ["user_age"])])
    interaction_pre = extend(
        history_pre,
        [
            ("device_x_time", PairwiseInteractionFeatures("device_type", "time_of_day"),
             ["device_type", "time_of_day"]),
            ("category_x_device", PairwiseInteractionFeatures("ad_category", "device_type"),
             ["ad_category", "device_type"]),
        ],
    )

    X_all, y_all = df[list(FEATURES)], df[TARGET]

    # Guard against the all-zero-block bug: a mis-named reindex produces a block of
    # zeros, the model silently ignores it, and the comparison looks like "interactions
    # add nothing" rather than "interactions were never there".
    age_block = AgeBandFeatures().fit(X_all)
    age_matrix = age_block.transform(X_all)
    check("age bands are not degenerate", age_matrix.sum() > 0,
          f"nonzero cells {int(age_matrix.sum())}")
    check("exactly one age band per row", np.allclose(age_matrix.sum(axis=1), 1))

    inter_block = PairwiseInteractionFeatures("device_type", "time_of_day").fit(X_all)
    inter_matrix = inter_block.transform(X_all)
    check("interactions are not degenerate", inter_matrix.sum() > 0,
          f"nonzero cells {int(inter_matrix.sum())}")
    check("exactly one interaction level per row", np.allclose(inter_matrix.sum(axis=1), 1))

    factories = {
        "raw (no derived)": lambda: assemble(build_preprocessor(engineer_features=False)),
        "+prior history": lambda: assemble(history_pre),
        "+prior history + age bands": lambda: assemble(age_pre),
        "+prior history + interactions": lambda: assemble(interaction_pre),
        "+prior history + select (l1)": lambda: assemble(
            history_pre, FeatureSelector(strategy="l1", k=12)
        ),
        "+prior history + select (MI)": lambda: assemble(
            history_pre, FeatureSelector(strategy="mutual_info", k=12)
        ),
    }

    comparison = evaluate_pipelines(X_all, y_all, factories, flag_rate=TARGET_FLAG_RATE)
    print(comparison.to_string(index=False))
    print("\n(repeated stratified 5-fold x10 on 19 positives; compare means against the std)")
    print("EDA single-rule baseline for scale: recall 0.80 at an 8.5% flag rate")

    print("\nWhich features survive selection, across folds:")
    stability = selection_frequency(
        lambda: assemble(history_pre, FeatureSelector(strategy="l1", k=12)),
        X_all,
        y_all,
    )
    stable = stability[stability["share_of_fits"] >= 0.8]
    print(stability.head(12).to_string(index=False))
    print(f"\nselected in >=80% of the 50 fits: {len(stable)} of {len(stability)} features")
    print(f"median feature is kept in {stability['folds_selected'].median():.0f} of 50 fits")

    rule("10. Baselines and the open-set question")

    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    from src.models import (
        AlwaysNoneClassifier,
        PriorHistoryRuleClassifier,
        cross_validate_models,
        evaluate_at_flag_rates,
        open_set_auc_by_category,
        open_set_report,
        scores_from_model,
    )

    # Diagnostic: everything before this point says the extra columns only add variance.
    # This model sees the prior-clicks block and nothing else, so the gap to the full
    # model is the cost of the cohort features.
    numeric_only = ColumnTransformer(
        [build_preprocessor().transformers[0]], remainder="drop", verbose_feature_names_out=True
    )

    baseline_factories = {
        "always none (floor)": lambda: AlwaysNoneClassifier(),
        "logistic (prior-clicks only)": lambda: Pipeline(
            [
                ("preprocess", numeric_only),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "prior-history rule": lambda: PriorHistoryRuleClassifier(cut=0.5),
        "logistic regression": lambda: build_model(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "random forest": lambda: build_model(
            RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=-1,
            )
        ),
        "hist gradient boosting": lambda: build_model(
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                random_state=RANDOM_STATE,
            )
        ),
    }

    folds, summary = cross_validate_models(
        X_all, y_all, baseline_factories, flag_rates=(0.05, 0.085), n_repeats=10
    )
    threshold_free = summary.drop_duplicates("model")[
        ["model", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std"]
    ].sort_values("pr_auc_mean", ascending=False)
    print("threshold-free:")
    print(threshold_free.to_string(index=False))

    operating = summary[summary["flag_rate_target"] == 0.085][
        ["model", "recall_mean", "recall_std", "precision_mean", "f1_mean",
         "achieved_flag_rate"]
    ].sort_values("recall_mean", ascending=False)
    print("\nat an 8.5% flag rate, threshold chosen on the training fold:")
    print(operating.to_string(index=False))
    print("\n(repeated stratified 5-fold x10, 19 positives; std is the honest column)")

    # One worked operating point, held out, for the story rather than the ranking.
    rule_model = PriorHistoryRuleClassifier(cut=0.5).fit(X_train, y_train)
    rule_scores = scores_from_model(rule_model, X_test)
    print("\nprior-history rule on the 191-row test split:")
    print(
        evaluate_at_flag_rates(
            y_test, rule_scores, flag_rates=(0.085,), threshold_scores=rule_scores
        )[["flag_rate_target", "tp", "fp", "fn", "precision", "recall", "roc_auc"]].to_string(index=False)
    )

    print("\nOpen-set behaviour, held-out ad_category as the unknown class")
    print("(none score = 1 - max predicted probability; higher = more none-like)")

    known_frames, unknown_results = {}, []
    for holdout in ["Gaming", "Travel", "Education"]:
        X_tr, X_te, y_tr, y_te, unknown_df = make_open_set_split(df, holdout)
        report, _ = open_set_report(
            X_tr, y_tr, X_te, unknown_df[list(FEATURES)], lambda: build_model(
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
            )
        )
        unknown_results.append(
            {
                "holdout": holdout,
                "n_unknown": report.n_unknown,
                "auroc": round(report.auroc, 3),
                "detection_at_5fpr": round(report.detection_rate_at_5fpr, 3),
                "known_mean": round(report.known_mean_none_score, 3),
                "unknown_mean": round(report.unknown_mean_none_score, 3),
            }
        )
    print(pd.DataFrame(unknown_results).to_string(index=False))
    print("\n0.5 AUROC is chance. These are 0.61-0.65 — barely above it, and at a 5% "
          "false-positive budget the score catches 3-8% of unknowns, so in practice it "
          "detects nothing. That is the gap Sprint 1 named, measured on three holdouts.")

    rule("Summary")

    print(f"rows used: {len(df)} | features in: {len(FEATURES)} | features out: {len(names)}")
    print(f"base positive rate: {positive_rate(df[TARGET]):.2%}")
    print(f"pipeline: {model}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
