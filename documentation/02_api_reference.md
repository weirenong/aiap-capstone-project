# API Reference

Six modules under `src/`, plus a scoring CLI. Signatures below are copied from the running
code, not from memory. Everything is importable from the repository root:

```python
from src.preprocessing import build_model, load_data
from src.models import cross_validate_models, PriorHistoryRuleClassifier
```

Conventions: `X` is a `pandas.DataFrame` whose columns match `FEATURES` unless stated
otherwise; `y` is a binary `pandas.Series` (1 = click, 0 = none). All transformers follow
scikit-learn's `fit` / `transform` contract and are safe to clone.

---

## `src.preprocessing`

Schema constants, loading, row filtering, and the composed preprocessing pipeline.

### Constants

| Name | Value |
|---|---|
| `TARGET` | `"clicked_ad"` |
| `NONE_LABEL` / `POSITIVE_LABEL` | `0` / `1` |
| `NUMERIC_FEATURES` | `("user_age", "previous_clicks_on_similar_ads")` |
| `CATEGORICAL_FEATURES` | `("gender", "device_type", "time_of_day", "ad_placement_position", "ad_category")` |
| `FEATURES` | numeric + categorical, in that order |
| `MIN_AGE` | `18` |
| `RANDOM_STATE` | `42` |
| `DEFAULT_DATA_PATH` | `<repo>/data/synthetic_data_243.csv` |

### `load_data(path=DEFAULT_DATA_PATH) -> DataFrame`

Reads the CSV and calls `validate_schema`. Raises `ValueError` on a missing column,
non-binary target, or empty frame.

### `validate_schema(df) -> None`

Checks required columns, binary target, numeric `user_age`, non-empty frame. Raises rather
than returning a boolean, so it can be used as an assertion at any entry point.

### `filter_eligible_rows(df, min_age=18) -> (DataFrame, dict)`

Drops rows below `min_age` and exact duplicates. **Must run before splitting** — dropping
rows inside a `Pipeline` desynchronises `X` and `y`.

Returns a report dict: `rows_in`, `under_min_age`, `duplicate_rows`, `rows_out`,
`positives_in`, `positives_out`, `positives_dropped`. On this dataset it removes 45 rows
and 1 positive.

### `build_preprocessor(*, engineer_features=True, clip_outliers=True, scale_numeric=True, numeric_imputer="median", categorical_imputer="most_frequent", add_missing_indicator=False, iqr_factor=1.5) -> ColumnTransformer`

Builds the two-block preprocessing column transformer:

```
numeric:     PriorHistoryFeatures → SimpleImputer → IQRClipper → StandardScaler
categorical: SimpleImputer → OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

Yields 24 columns from 7. `add_missing_indicator` is off because this file has no nulls.

### `build_model(estimator, *, engineer_features=True, clip_outliers=True, scale_numeric=True, iqr_factor=1.5) -> Pipeline`

Preprocessor + estimator as one fitted object (`steps: preprocess, model`). Use this rather
than composing by hand — it is what guarantees the transforms cannot see held-out data.

### `build_resampled_model(estimator, sampler, **preprocess_kwargs) -> Pipeline`

Same, but with a sampler (SMOTE or similar) between preprocessing and the estimator, using
`imblearn.pipeline.Pipeline`. Raises `ImportError` with an actionable message if
`imbalanced-learn` is absent. Kept separate on purpose: resampling must not become a
preprocessing step you can call before the split.

```python
from imblearn.over_sampling import SMOTE
pipe = build_resampled_model(LogisticRegression(class_weight="balanced"), SMOTE())
```

### `split_features_target(df) -> (X, y)`

Feature frame and `int` target, none label untouched.

### `make_train_test_split(df, test_size=0.2, random_state=42) -> (X_train, X_test, y_train, y_test)`

Stratified. Non-negotiable at a 2% positive rate.

### `make_open_set_split(df, holdout_category, test_size=0.2, random_state=42) -> (X_train, X_test, y_train, y_test, unknown_df)`

Holds out an entire `ad_category` as the unknown class and returns it separately. The
returned `unknown_df` still contains the target column for analysis. Raises `ValueError`
for an unknown category. Underpowered at 19 positives — use as a behavioural check, not a
headline metric.

### `positive_rate(y) -> float`

Share of rows with label 1.

### `transform_frame(preprocessor, X) -> DataFrame`

Fitted-transform to a labelled DataFrame; for inspection, not modelling.

### `IQRClipper(factor=1.5)`

Clips numeric columns to training quantile fences.

**Gotcha:** columns with zero IQR are left unclipped. Without that rule a binary flag gets
clipped to a constant, because its 25th and 75th percentiles are both 0 at a 9% base rate.
This was a real bug, now pinned by a check in `main.py`.

---

## `src.feature_engineering`

Feature creation, in-fold selection, and selection diagnostics.

### `PriorHistoryFeatures(column="previous_clicks_on_similar_ads", low_history_cut=0.5, add_log=True)`

Adds `<column>__log1p` and `<column>__low_history`. The cut is a fixed constant derived
from EDA, not a fitted parameter — see the caveat in its docstring.

### `AgeBandFeatures(column="user_age", bins=(0,25,35,45,55,120), labels=("<25","25-34","35-44","45-54","55+"))`

One-hot age bands. Reindexes to the fitted category list, so a band absent from a fold
still produces a zero column.

### `PairwiseInteractionFeatures(left, right, separator="|")`

One-hot cross of two categorical columns. Unseen combinations encode to all zeros.

**Gotcha:** `get_dummies` labels columns with the bare level; the transformer renames them
before reindexing. Skipping that step silently produces an all-zero block — also a real
bug, also pinned by a check.

### `FeatureSelector(strategy="l1", k=12, random_state=42)`

In-pipeline selection. `strategy="mutual_info"` keeps top `k`; `strategy="l1"` keeps
non-zero L1 coefficients (threshold `"median"`). Exposes `support_`, so callers can recover
the selected names.

### `mutual_information_table(X, y, top_n=15) -> DataFrame`

Global MI ranking, for reference. Unstable at this sample size; report it next to
`selection_frequency`, never alone.

### `selection_frequency(pipeline_factory, X, y, *, selector_step="select", n_splits=5, n_repeats=10, random_state=42) -> DataFrame`

How often each feature survives across `n_splits × n_repeats` fits. Requires the pipeline
to have a step named `selector_step` holding a `FeatureSelector`. Columns: `feature`,
`folds_selected`, `share_of_fits`, `n_fits`. Handles the case where the selector receives a
numpy array by pulling real names from the `preprocess` step.

### `evaluate_pipelines(X, y, factories, *, n_splits=5, n_repeats=10, flag_rate=0.085, random_state=42) -> DataFrame`

CV comparison returning `pr_auc_mean`, `pr_auc_std`, `recall_mean`, `recall_std`,
`n_features` per named factory. Thresholds come from the training fold each time.

---

## `src.models`

Baselines, metrics, CV comparison, and the first-pass open-set report.

### `AlwaysNoneClassifier()`

The floor. `predict` returns zeros; `predict_proba` returns `[1, 0]`. Use it to prove that
a metric is not just measuring the majority class.

### `PriorHistoryRuleClassifier(column="previous_clicks_on_similar_ads", cut=0.5)`

`predict` = `column <= cut`. `decision_function` = `cut - column` (higher = more likely to
click), which gives a valid ranking for PR-AUC. `predict_proba` is a monotone squashing and
is **not calibrated** — do not report Brier for this model.

### `scores_from_model(model, X) -> ndarray`

`predict_proba[:, 1]` when available, else `decision_function`.

### `threshold_at_flag_rate(scores, flag_rate) -> float`

Highest threshold that flags at most `flag_rate` of the rows it is computed on.

**Gotcha:** call it with *training* scores. Two tie rules matter: walking distinct values
ascending gives the most recall inside the budget, and a constant score returns a threshold
above the maximum so nothing is flagged. Naive `np.quantile` makes the always-none floor
look like a perfect detector.

### `confusion_at(y_true, scores, threshold) -> dict`

`tp, fp, fn, tn, precision, recall, f1, flag_rate, false_positive_rate, specificity`.

### `ranking_metrics(y_true, scores, probabilities=None) -> dict`

`n, positives, roc_auc, pr_auc, brier`. Returns `NaN` for the ranking metrics when a class
is absent rather than raising.

### `evaluate_at_flag_rates(y_true, scores, *, flag_rates=(0.05,0.085,0.10), threshold_scores=None, probabilities=None) -> DataFrame`

One row per operating point. `threshold_scores` are the scores used to choose thresholds
(training fold); defaulting them to the evaluation scores is only correct for a final
held-out report.

### `cross_validate_models(X, y, factories, *, flag_rates=(0.05,0.085,0.10), n_splits=5, n_repeats=10, random_state=42) -> (folds, summary)`

Repeated stratified CV. Returns per-fold rows (`tp/fp/fn/tn`, metrics, threshold, achieved
flag rate) and a mean/std summary including `brier_mean`.

### `none_score(model, X) -> ndarray`

`1 - max(predict_proba)`; higher means more none-like. Retained for comparison — see
`src.advanced_models` for the detector that actually works.

### `OpenSetReport` (dataclass)

Fields: `auroc`, `aupr`, `detection_rate_at_5fpr`, `known_mean_none_score`,
`unknown_mean_none_score`, `n_known`, `n_unknown`, `threshold_at_5fpr`. `as_dict()` for
serialisation.

### `open_set_report(known_train_X, known_train_y, known_test_X, unknown_X, factory) -> (OpenSetReport, fitted_model)`

Fits on known classes only, then treats unknown rows as positives for detection. AUROC is
defensible here because the unknown set supplies the positives, not the 19 click labels.

### `open_set_auc_by_category(known_train_X, known_train_y, known_test_X, unknown_frames, factory) -> DataFrame`

Runs the report per held-out category. Prefer this over a single holdout — semantic shift
dominates open-set difficulty.

---

## `src.advanced_models`

Distinct architectures and a proper set of none detectors.

### `svm_pipeline(preprocessor=None, *, C=1.0, gamma="scale")`

RBF SVM with `class_weight="balanced"` and `probability=True` (internal Platt). The
calibration cost is visible as a very low Brier score combined with mediocre ranking.

### `mlp_pipeline(preprocessor=None, *, hidden_layer_sizes=(32,16), alpha=1e-3, max_iter=800)`

Small MLP with early stopping. Included as a capacity stress test; it underfits here.

### `knn_pipeline(preprocessor=None, *, n_neighbors=15)`

Distance-weighted k-NN. Its probabilities are heavily tied, which breaks threshold
calibration at a fixed flag rate — check `achieved_flag_rate` before comparing it.

### `energy_from_margin(margin) -> ndarray`

`-logaddexp(0, margin)`. In binary problems this is monotone in the margin, so it ranks
identically — a point the detector comparison demonstrates empirically.

### `NoneDetector(method="mahalanobis", k=5)`

Fitted on the **preprocessed** design matrix. `fit(X)` then `score(X) -> ndarray`, higher =
more none-like. `method="mahalanobis"` uses a shrunken pooled covariance (10% shrinkage,
pseudo-inverse); `method="knn"` uses the distance to the k-th nearest reference row.

### `transformation_none_score(model, X, kind="maxprob")`

`kind` ∈ `{"maxprob", "margin", "energy"}`. Scores derived from the classifier head.

### `none_detector_comparison(known_train_X, known_train_y, known_test_X, unknown_X, *, classifier_factory=svm_pipeline) -> DataFrame`

AUROC, AUPR, detection at 5% FPR, and mean scores for all five detectors on one unknown
set. Note that Mahalanobis and k-NN depend only on the preprocessed features, so they come
out identical across different classifiers fitted with the same preprocessing.

### `paired_comparison(folds, reference, metric="pr_auc") -> DataFrame`

Per-fold deltas against a named model. Columns: `model, mean_delta, std_delta, wins,
folds, win_rate`. Requires the same folds for every model.

---

## `src.model_optimization`

Nested CV, tuning diagnostics, and final selection.

### `search_spaces(include_selector=False) -> dict[str, dict]`

Small grids per model, prefixed for pipeline steps (`model__C`, `select__k`). Deliberately
narrow: a wide grid on 19 positives searches until something fits the fold.

### `model_factories(include_selector=False) -> dict[str, callable]`

Factories matching the space names: `logistic regression`, `random forest`, `rbf svm`,
`knn`, `mlp`, optionally `logistic + select`. Each call builds a fresh preprocessor.

### `nested_cv(X, y, *, model_names=None, outer_splits=5, outer_repeats=2, inner_splits=3, flag_rate=0.085, reference_factory=None, random_state=42) -> (fold_rows, param_rows)`

Tuning happens inside each outer training split; the outer fold is scored once. Passing
`reference_factory` (normally the rule) scores it on the same outer folds, so the margin is
paired.

`fold_rows` columns include `tuned_pr_auc`, `default_pr_auc`, `inner_best_score`,
`reference_pr_auc`, `tuned_recall`, `n_test_positives`. `param_rows` records
`model, fold, parameter, value` for every selected parameter.

### `summarize_nested(folds) -> DataFrame`

Per-model means and the paired tuning gain, plus `outer_minus_inner`: negative means the
outer score came in below the inner estimate, i.e. the search overfit the inner folds.

### `tuning_stability(params) -> DataFrame`

`model, parameter, value, selected, folds, share`. Anything under ~0.8 is a coin flip and
should not be described as "the tuned setting".

### `learning_curve_report(factory, X, y, *, train_sizes=(20,40,60,80,100), cv_splits=5, random_state=42) -> DataFrame`

Train vs cross-validated PR-AUC with the gap. Sizes are absolute row counts because
fractions hide how few positives each point contains.

**Gotcha:** with 1–2 positives per validation fold, average precision can only take a few
discrete values. `std = 0.000` at small sizes is granularity, not stability.

### `select_final_model(summary, *, reference_pr_auc, reference_name="prior-history rule", tie_tolerance=0.02) -> dict`

Returns `selected`, `reason`, `margin_over_reference`, `decisive`. A tuned model must beat
the reference by more than the tolerance **and** more than its own fold-to-fold std;
otherwise the reference wins because it has no fitted parameters to go stale.

---

## `src.artifacts`

Persistence.

### `new_run_dir(base=RESULTS_ROOT, label="run") -> Path`

Timestamped directory with a collision suffix.

### `list_runs(base=RESULTS_ROOT) -> list[Path]` / `load_latest_run(base=RESULTS_ROOT) -> Path | None`

`load_latest_run` prefers `latest.json` and falls back to the newest directory.

### `save_frame(frame, name, run_dir) -> dict` / `save_json(payload, name, run_dir) -> str` / `save_model(model, name, run_dir) -> str`

Every table is written as both CSV and JSON. Payloads may contain numpy scalars,
dataclasses and paths; they are normalised before serialisation, and `NaN` becomes `null`.

### `load_frame(run_dir, name)` / `load_json(run_dir, name)` / `load_model(run_dir, name)`

The read side. Use these rather than hand-built paths so the layout can change in one place.

### `file_digest(path) -> dict`

`path`, `bytes`, `sha256`. Recorded in every manifest so a result can be tied to exact
input bytes.

### `write_manifest(run_dir, *, config=None, data_path=None) -> dict`

Records `run_id`, `created_utc`, `git_commit`, package versions, config and the data digest.

### `write_pointer(run_dir, headline, base=RESULTS_ROOT) -> str` / `write_index(run_dir, summary) -> str`

Write `results/latest.json` and a human-readable `index.md`.

### `compare_runs(run_a, run_b, metric="pr_auc_mean") -> DataFrame`

Joins two `cv_summary` tables on model and flag rate and returns the deltas. Comparing runs
should be a read, not a re-run.

---

## `src.predict`

Batch scoring CLI. Not an importable API beyond `score()`.

```bash
python -m src.predict --input rows.csv [--run results/run_x] [--output scored.csv]
                      [--threshold 0.30] [--drop-target]
```

| Flag | Meaning |
|---|---|
| `--input` | CSV to score; must contain all seven feature columns |
| `--run` | run directory; defaults to the latest |
| `--output` | write scored CSV; otherwise summary only |
| `--threshold` | override the threshold recorded in `scoring.json` |
| `--drop-target` | drop `clicked_ad` if present |

Output columns: the input columns plus `p_click` (6 dp) and `flagged` (0/1).

Exit codes: `0` success, `2` bad input (missing columns or empty), `3` missing run
artifacts.

### `score(frame, model, threshold) -> DataFrame`

The function the CLI wraps, if you want to score inside Python.
