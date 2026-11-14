# Deployment Guide

How to install, regenerate, score, and operate this project. Everything below is exercised
by `python main.py` or by the commands shown, so a failure here is a real failure.

## 1. Requirements

- Python 3.12 (the pinned versions were resolved on 3.12.2)
- No GPU, no services, no database
- ~50 MB for the environment, ~2 MB for the repository including artifacts

```bash
git clone https://github.com/weirenong/aiap-capstone-project.git
cd aiap-capstone-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins exact versions. Do not float them: the persisted model is a joblib
pickle and will warn or fail across scikit-learn minor versions.

## 2. Two entry points

| Command | Purpose | Runtime |
|---|---|---|
| `python main.py` | Full harness: validates every stage, trains all models, writes a run directory | ~2.5 min |
| `python -m src.predict` | Batch scoring with a persisted run | seconds |

`main.py` is a verification harness, not a demo. It raises on failure, so its exit code is
the signal — `0` means every assertion passed.

## 3. Producing a deployable artefact

```bash
python main.py
cat results/latest.json          # run id + headline metrics
ls results/$(ls results | grep run_ | tail -1)
```

The deployable pair is:

- `baseline_logistic.joblib` — fitted preprocessor + classifier, one object
- `scoring.json` — threshold, target flag rate, feature list, model type, provenance

Both are copied into every run directory. `scoring.json` is written alongside the model
precisely so nobody has to guess a threshold; `0.5` would flag almost nothing at a 2% base
rate.

## 4. Batch scoring

```bash
python -m src.predict \
  --input data/new_impressions.csv \
  --output scored.csv
```

```
run:        .../results/run_20261114_043902
model:      baseline_logistic (logistic regression, class_weight=balanced, ...)
data sha256:a4be6e2324f172d3...
rows:       8
threshold:  0.7575 (target flag rate 8.5%)
flagged:    1 (12.5%)
score range:0.0027 - 0.8817
written:    scored.csv
```

### Input contract

CSV, UTF-8, header row. Required columns, exact names:

| Column | Type | Notes |
|---|---|---|
| `user_age` | numeric | Rows under 18 should be filtered upstream; the model saw none |
| `previous_clicks_on_similar_ads` | numeric | Continuous; fractional values are expected |
| `gender` | string | Unseen values are tolerated |
| `device_type` | string | as above |
| `time_of_day` | string | as above |
| `ad_placement_position` | string | as above |
| `ad_category` | string | Unseen values encode to all-zero — see "New categories" |

`clicked_ad` is not a feature. If it is present, pass `--drop-target`; otherwise the column
is simply carried through to the output unscored.

Missing columns exit with code 2 and a list of what is missing. An empty file also exits 2.

### Output contract

The input columns, plus:

| Column | Type | Meaning |
|---|---|---|
| `p_click` | float, 6 dp | Ranking score. **Not a calibrated probability** |
| `flagged` | int 0/1 | `p_click >= threshold` |

Rows keep their original order. No rows are dropped.

### Choosing a threshold

`--threshold` overrides the stored value. Change it when the *cost* changes, not when the
model changes:

```bash
python -m src.predict --input rows.csv --threshold 0.30    # wider net, more false positives
```

The stored threshold corresponds to an 8.5% flag rate: about 1 in 7 flags converts, and
roughly 42% of clicks are missed. Raising the rate buys recall at the cost of precision,
roughly linearly at this scale.

## 5. New categories and the None problem

Unseen `ad_category` values are encoded as an all-zero block rather than rejected, so the
model will still produce a confident `p_click` for an unfamiliar category. That is the
wrong behaviour if a new category should be treated as unknown.

Use the distance detector for that decision. It is not wrapped in the CLI yet; the
reference implementation is `src.advanced_models.NoneDetector`:

```python
from src.artifacts import load_model, load_latest_run
from src.advanced_models import NoneDetector

run = load_latest_run()
model = load_model(run, "baseline_logistic")
pre = model.named_steps["preprocess"]
detector = NoneDetector("mahalanobis").fit(pre.transform(training_frame))
score = detector.score(pre.transform(new_frame))     # higher = more none-like
```

Fitted on training data only. Reported performance: AUROC 0.864–0.890 on held-out
categories, 20–27% detection at a 5% false-positive budget.

## 6. Monitoring

Nothing is instrumented, but these are the signals worth building, in order:

1. **Input drift.** Mean/quantiles of `previous_clicks_on_similar_ads` and the share of
   each category. A shift here invalidates the threshold before it invalidates the model.
2. **New-category rate.** Unseen `ad_category` values per batch. Above a few percent, the
   detection path is needed, not the classifier.
3. **Flag rate versus budget.** Compare `flagged.mean()` to the 8.5% design target per
   batch. A persistent gap means the score distribution moved.
4. **Delayed conversion rate among flags.** The only real measure of precision. Needs a
   join to outcomes with a lag; expect it to drift first when the generator or audience
   changes.

There is no ground truth available at scoring time, so a batch metric cannot be computed
immediately. Any retraining trigger has to be driven by drift proxies plus the delayed
label join.

## 7. Retraining

```bash
# new data in place of data/synthetic_data_243.csv, then:
python main.py
```

Compare the new run to the previous one before promoting it:

```python
from src.artifacts import compare_runs, list_runs
runs = list_runs()
diff = compare_runs(runs[-2], runs[-1], metric="pr_auc_mean")
print(diff.sort_values("delta", ascending=False))
```

Promotion rule used in this project: the new run must beat the previous one by more than
the fold-to-fold standard deviation on PR-AUC, and must not regress calibration or the
open-set AUROC. Anything inside the noise band is a tie, and the simpler artefact wins.

## 8. Rollback

Every run is a self-contained directory. Rollback is repointing `results/latest.json` — or,
if the pointer is managed externally, passing `--run` explicitly:

```bash
python -m src.predict --input rows.csv --run results/run_<previous>
```

No state lives outside the run directory, so there is nothing else to undo.

## 9. Known failure modes

| Symptom | Cause | Action |
|---|---|---|
| `KeyError` on a feature | Schema drift between producer and model | Check `manifest.json` → `config.features`; fix at source |
| Exit code 2 | Missing/empty input | Read the listed missing columns |
| Exit code 3 | No run or missing artifacts | `python main.py`, or pass `--run` |
| All probabilities near 0.9, few flags | Threshold mismatch | Confirm `scoring.json` is from the same run as the model |
| `p_click` used as a probability | Balanced class weights inflate scores | Calibrate, or use ranking only — see the model card |
| Unpickling warning | scikit-learn version drift | Reinstall from `requirements.txt`; refit rather than unpickle across minors |
| New category silently scored | `handle_unknown="ignore"` | Route through `NoneDetector` |

## 10. What this deployment does not include

No API service, no scheduler, no authentication, no logging pipeline, no container image.
The intent is a reproducible batch artefact, not a production service. Wrapping the CLI in
a job runner is straightforward; deciding to expose predictions to an automated bidding
system is not, and the precision figures above are the reason.
