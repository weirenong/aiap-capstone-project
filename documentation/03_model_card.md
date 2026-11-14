# Model Card: Ad Click Classifier ("None" framing)

## Model details

| Item | Value |
|---|---|
| Model type | Logistic regression (`class_weight="balanced"`) inside a scikit-learn pipeline |
| Version | run `20261114_043902` (see `results/latest.json` for the current one) |
| Inputs | 7 columns: `user_age`, `previous_clicks_on_similar_ads`, `gender`, `device_type`, `time_of_day`, `ad_placement_position`, `ad_category` |
| Outputs | `p_click` (probability), `flagged` (1 if `p_click ≥ threshold`) |
| Threshold | Run-recorded, chosen at the training split's 8.5% quantile (≈0.758) |
| Trained on | 955 rows, 19 positives, seed 42 |
| Artefact | `baseline_logistic.joblib` + `scoring.json` in the run directory |
| Owner | Capstone project, AIAP cohort |

The persisted model is the **deployable pipeline**, not the best-performing model. See
"Performance" below — a one-line rule beats it.

## Intended use

**Intended:**

- Ranking impressions by predicted click propensity for prioritisation, at a fixed
  impression budget.
- Demonstrating a leakage-safe, reproducible imbalanced-classification workflow.
- Batch scoring where a probability ordering is useful and a human reviews the output.

**Not intended:**

- Fully automated bidding or budget allocation. The precision is far too low.
- Any use where a missed click or a wasted impression has material cost and the ~17%
  precision has not been priced in.
- Individual-level decisions about users, especially price discrimination.
- Predicting behaviour of minors — those rows were removed from training and should be
  filtered from inference input.

## Training data

Synthetic ad-impression data (`data/synthetic_data_243.csv`), 1,000 rows. After removing 45
under-18 records: 955 rows, 19 positives (2.0%). The generator produces a near-deterministic
relationship between `previous_clicks_on_similar_ads` and the label, which is the main
reason a rule outperforms learned models. **This is a property of the file, not of ad
traffic in general**, and the model card would need rewriting before use on real data.

## Factors

The evaluation slices the data by `ad_category` (the only categorical with a detectable
effect) and uses held-out categories for the open-set analysis. Gender, device, time of day
and placement showed no effect distinguishable from noise at this sample size, so no
fairness conclusions can be drawn from them in either direction.

## Performance

Measured with repeated stratified 5-fold ×10 cross-validation, all 19 positives.

| Metric | Value | Reference |
|---|---|---|
| PR-AUC | 0.258 ± 0.150 | rule 0.390; floor 0.020 |
| ROC-AUC | 0.822 ± 0.099 | rule 0.913 |
| Recall @8.5% flag rate | 0.578 ± 0.219 | rule 0.702 |
| Precision @8.5% flag rate | 0.141 | rule 0.172 |
| Brier | 0.095 | floor 0.020 — **worse** |
| Accuracy | not reported | floor achieves 98% |

Interpretation: at the operating point, roughly one in seven flagged impressions converts
against a 2.0% base rate — about a 7× lift — and about 42% of clicks are missed. Five of
every six flags are false positives.

## Calibration

Badly calibrated, and this matters. `class_weight="balanced"` reweights the training
distribution, so `p_click` does not estimate the true 2% rate; the always-none model has a
five-times better Brier score despite being useless. **Do not use `p_click` as a
probability** without post-hoc calibration (isotonic or Platt on a validation split). Use
it as a ranking score, or apply the recorded threshold, which was chosen with this
distortion in mind.

## Open-set behaviour

The model has no mechanism for saying "none of the above". Given a category it has never
seen, `handle_unknown="ignore"` encodes it as an all-zero block and the classifier still
returns a confident known-class prediction.

Detection of a held-out category with the classifier head's own confidence: AUROC 0.572–0.645.
With a Mahalanobis distance on the preprocessed features: **0.864–0.890**, catching 20–27%
of unknown rows inside a 5% false-positive budget. If the deployment needs "is this input
new?", use the distance detector, not `p_click`.

Caveat: much of that signal comes from the category one-hot block going all-zero. It will
not transfer to continuous-only features.

## Ethical considerations

- **Minors.** 45 rows were under 18 (minimum 6.99). They were excluded from training, and
  inference input should be filtered the same way. Removing them cost one positive — a
  deliberate trade.
- **Persuasion and manipulation.** Click propensity is not consent, and a model that
  targets susceptible users raises questions that accuracy metrics cannot answer.
- **Feedback loops.** A deployed model changes what impressions are shown, which changes
  future training data. Nothing in this project addresses that.
- **Privacy.** No identifiers beyond coarse demographics; no attempt to re-identify.
- **Synthetic data.** No real users are represented, which limits both the ethical surface
  and the external validity.

## Caveats and recommendations

1. **The rule wins.** Deploying this model instead of `previous_clicks ≤ 0.5` would lose
   PR-AUC (0.258 vs 0.390). The honest deployment is the rule plus the distance detector,
   with the model as a fallback or ensemble component.
2. **Nineteen positives.** Every number carries a wide interval; a single click moves recall
   by 5 percentage points. Differences under ~0.10 PR-AUC are not established.
3. **No monitoring in this repo.** Drift detection, retraining triggers and performance
   tracking are described in the deployment guide but not implemented.
4. **Re-validate before reuse.** If the data changes, rerun `python main.py`; the manifest
   records the data hash, so a mismatch is detectable rather than silent.
