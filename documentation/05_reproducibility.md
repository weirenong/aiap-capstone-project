# Reproducibility

## 1. What "reproducible" means here

Three separate claims, each with its own evidence:

1. **The environment can be rebuilt.** Exact pins in `requirements.txt`.
2. **The results repeat.** Same code, same data, same seed → identical metrics. Verified by
   running the harness twice and diffing the summaries: maximum absolute delta `0.0`.
3. **A run explains itself.** Each run directory records the commit, versions, config, and
   the **sha256 of the input data**, so a result can be tied to the exact bytes that
   produced it.

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pinned: `pandas 2.2.1`, `numpy 1.26.4`, `scikit-learn 1.4.2`, `scipy 1.17.1`,
`imbalanced-learn 0.12.3`, `joblib 1.6.0`, `matplotlib 3.8.4`, `seaborn 0.13.2`,
`ipykernel 7.4.0`. Python 3.12.2 was used.

scikit-learn minor versions are not interchangeable for the persisted `.joblib` model:
unpickling across versions emits warnings and can fail. Refit rather than cross-load.

## 3. Seeds

| Location | Seed |
|---|---|
| `RANDOM_STATE` in `src.preprocessing`, `src.models`, `src.advanced_models`, `src.model_optimization` | 42 |
| Split | `train_test_split(..., stratify=y, random_state=42)` |
| CV | `RepeatedStratifiedKFold(..., random_state=42)` |
| Inner tuning CV | `StratifiedKFold(shuffle=True, random_state=42)` |
| Random forest / MLP / SVM | model-level `random_state=42` |
| EDA notebook | `RANDOM_STATE = 42` |

Anything unseeded would show up as a non-zero delta between two runs; it does not.

## 4. Verifying a run

`main.py` performs 42 assertions and exits non-zero on any failure. The ones that matter
for reproducibility:

| Check | What it proves |
|---|---|
| IQR fences equal train-only fences | preprocessing did not see the full frame |
| Scaler means equal clipped train means | no full-data statistics |
| Unseen category → all-zero block | documented inference behaviour holds |
| Two fits give identical output | no hidden randomness in the pipeline |
| `cv_folds` reloads with the same row count | persistence round-trips |
| Reloaded model predicts identically | the artefact is the model, not a lookalike |
| `latest.json` resolves to this run | discovery works |
| `compare_runs(run, run)` returns zeros | the comparison helper is correct |


Re-verify by hand:

```python
from src.artifacts import list_runs, compare_runs
runs = list_runs()
print(float(compare_runs(runs[-2], runs[-1])["delta"].abs().max()))   # expect 0.0
```

## 5. What the manifest records

```json
{
  "run_id": "run_20261114_043902",
  "created_utc": "2026-11-14T04:39:02+00:00",
  "git_commit": "…",
  "versions": {"python": "3.12.2", "pandas": "2.2.1", "numpy": "1.26.4",
               "scikit-learn": "1.4.2", "joblib": "1.6.0",
               "imbalanced-learn": "0.12.3"},
  "config": {"target": "clicked_ad", "features": ["…"], "flag_rates": [0.05, 0.085, 0.1],
             "cv": {"n_splits": 5, "n_repeats": 10}, "random_state": 42,
             "low_history_cut": 0.5, "min_age": 18},
  "data": {"path": "data/synthetic_data_243.csv", "bytes": 55031,
           "sha256": "a4be6e23…"}
}
```

If `data.sha256` no longer matches the file on disk, the run describes different data. That
is the check to make before comparing two runs.

## 6. Regenerating the EDA notebook

```bash
jupyter nbconvert --to notebook --execute --inplace eda.ipynb
```

The notebook is committed with outputs, so it renders without execution. Re-running will
reproduce the same numbers; `figures` may differ in layout across matplotlib versions but
the statistics will not.

## 7. Comparing two runs

```python
from src.artifacts import compare_runs

diff = compare_runs("results/run_A", "results/run_B", metric="pr_auc_mean")
print(diff.sort_values("delta", ascending=False).head())
```

Matched on `model` and `flag_rate_target`, so a run that added a model shows only the
models both runs have.

## 8. Known non-determinism

- **Runtime**, obviously.
- **Directory names** (`run_<timestamp>`), which is why `latest.json` exists.
- **Parallel reductions** inside scikit-learn can reorder floating-point additions. The
  observed delta is 0.0 across the two retained runs on this machine; a different BLAS
  build may produce last-digit differences. Anything beyond the last digit is a real change.
- **`n_jobs=-1`** is used for random forests. Results are seeded, but thread count can
  affect summation order.

## 9. Common ways to break reproduction

| Change | Effect |
|---|---|
| Editing the CSV in place | `data.sha256` mismatch; all cached comparisons become invalid |
| Upgrading scikit-learn | pickled model may not load; refit needed |
| Changing `RANDOM_STATE` | CV folds change, all metrics move within their std |
| Re-running `main.py` | creates a new run; the old one is untouched, `latest.json` moves |
| Deleting `results/` | numbers in the report cannot be re-derived without re-running (~2.5 min) |
| Running only part of `main.py` | persistence checks in the final section are skipped, so a partial run looks successful |

## 10. Cost of a full regeneration

About 2.5 minutes on a laptop: roughly 65s for the baseline and open-set sections, 50s for
nested CV, and the rest in the advanced model comparison. The dominant cost is repeated
cross-validation (50 folds × up to 8 models), not tuning.
