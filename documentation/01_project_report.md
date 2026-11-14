# Project Report: None Classification for Ad Click Prediction

## 1. Summary

This project predicts whether an ad impression results in a click, framed as a **"None"
classification problem**: the system must be able to say "none of the known classes"
rather than forcing every input into one. In this dataset that takes two forms:

- **Background class.** 98% of impressions are non-clicks. A model that always predicts
  none is 98% accurate and worthless, so the project is about the 2% and about keeping
  the false-positive rate usable.
- **Open-set.** A category never seen in training should be flagged as none-of-the-above
  rather than confidently assigned to a known class.

The headline result is uncomfortable and consistent across every sprint: **a one-line
threshold rule outperforms every trained model on click ranking** (PR-AUC 0.390 vs 0.310
for the best tuned classifier). The more useful result is in the open-set half: a
Mahalanobis distance detector separates held-out ad categories at AUROC 0.864–0.890, where
softmax-based scores sit at 0.336–0.645.

## 2. Problem framing

| Item | Value |
|---|---|
| Target | `clicked_ad` (0 = none, 1 = click) |
| Rows | 1,000 raw, 955 after cleaning |
| Positives | 19 (2.0%) |
| Features | 2 numeric, 5 categorical |
| Primary metric | PR-AUC (threshold-free) and recall at a fixed flag rate |
| Secondary metrics | ROC-AUC, Brier, precision/F1 at the operating point |
| Deliberately unused | accuracy (98% for a constant predictor) |

Three framings were considered and the first two retained:

1. **Binary none-vs-click.** The dataset as given. This is the deployed task.
2. **Open-set recognition.** `ad_category` as the class space; hold one category out and
   test whether it is treated as unknown. Implemented as an evaluation, not a training
   objective, because 19 positives cannot support a dedicated none head.
3. **Multiclass category prediction.** Rejected: the data has no per-category engagement
   label, only a binary click, so this would have required inventing the target.

## 3. Data audit

| Check | Result | Consequence |
|---|---|---|
| Missing values | None | No imputation workstream; imputer kept in the pipeline for safety |
| Duplicate rows | None | No dedup step |
| Duplicate feature rows | 0 (1,000 unique) | Every row is distinguishable; no leakage from duplicates |
| `user_age` range | 6.99 – 69.32 | **45 rows under 18 removed** (costs 1 positive) |
| `previous_clicks_on_similar_ads` | 0.048 – 9.93, 99.4% fractional | Continuous, despite the count-like name |
| Class balance | 1 : 49 | Class weights first; resampling only inside folds |

The under-18 filter is a judgement call with a visible cost: removing 45 rows removes
**1 of 20 positives**, i.e. 5% of the minority class. The removal is defensible on
legal/ethical grounds for ad data, and it is applied once before splitting so the test
population matches the deployed population.

## 4. Feature analysis

| Feature | Evidence | Verdict |
|---|---|---|
| `previous_clicks_on_similar_ads` | Clickers mean 0.45, non-clickers 1.96 (p ≈ 0); click rate 15.8% below 0.5 and **0 of 391 rows above 2** | Dominant; a step function, not a line |
| `ad_category` | χ² = 22.76, p = 0.0004; Gaming 7.0%, Travel/Education 4.0%, others ≤0.8% | Second-strongest; used as the held-out class for open-set tests |
| `gender` | p = 0.12; 6 clicks in 480 women, 0 in 40 "Other" | Borderline; not separable from noise at this size |
| `device_type`, `time_of_day`, `ad_placement_position` | p > 0.5 | No detectable effect |
| `user_age` | p = 0.63, MI 0.000 | No linear or monotone effect; age bands tested and rejected |

Two cautions carried forward from the analysis:

- **Mutual information is unstable here.** It ranks `ad_category_Gaming` at 0.000 despite
  Gaming having the highest click rate, and moves with the random seed. It is used only to
  confirm that one feature dominates, never as a ranking.
- **The direction is counter-intuitive.** More prior clicks is associated with *not*
  clicking. In synthetic data this most likely reflects the generator rather than user
  behaviour, and it is reported as a property of the dataset, not interpreted.

## 5. Preprocessing

The pipeline is a single scikit-learn `Pipeline` so that preprocessing and model travel
together and every fitted statistic belongs to the fold it was fitted on.

```
numeric block (2 columns):
    PriorHistoryFeatures  →  SimpleImputer(median)  →  IQRClipper(1.5)  →  StandardScaler
categorical block (5 columns):
    SimpleImputer(most_frequent)  →  OneHotEncoder(handle_unknown="ignore")
```

Design decisions and their reasons:

| Decision | Reason |
|---|---|
| Row filtering outside the pipeline | Dropping rows inside a `Pipeline` desynchronises `X` and `y` |
| Impute before clip before scale | NaNs poison quantiles; scaling is applied to the values the model sees |
| `handle_unknown="ignore"` | An unseen category becomes an all-zero block — evidence against known classes |
| IQR clipping, never row deletion | At 19 positives, deleting outliers is not affordable |
| No target encoding | Removes a whole class of leakage at this sample size |

**A bug worth recording:** `IQRClipper` initially clipped the derived low-history flag to
zeros. A ~9% base rate puts both the 25th and 75th percentiles at 0, so the fences were
0 and 0. Zero-IQR columns are now left unclipped, and a regression check pins it.

## 6. Feature engineering and selection

Three builders were implemented and evaluated: prior-history features (log1p, low-history
flag), age bands, and categorical interactions. Six feature sets were compared under
repeated stratified CV:

| Feature set | n | PR-AUC | Recall @8.5% |
|---|---|---|---|
| + prior history + select (MI) | 12 | 0.271 ± 0.137 | 0.62 |
| + prior history + age bands | 29 | 0.264 ± 0.161 | 0.58 |
| **+ prior history** | 24 | 0.258 ± 0.148 | 0.59 |
| + prior history + interactions | 53 | 0.242 ± 0.160 | 0.55 |
| + prior history + select (L1) | 12 | 0.241 ± 0.133 | 0.60 |
| raw | 22 | 0.230 ± 0.150 | 0.49 |

Only the prior-history features clear the noise floor, and only they have a mechanism: the
EDA relationship is a step, and a linear model cannot represent a step without a flag.

Selection was kept as a **diagnostic**, not a pipeline step, because of the stability
result: 8 of 24 features survive ≥80% of 50 fits, and the three most stable are
`device_type_Tablet`, `gender_Other` and `ad_placement_position_Footer` — cells with
**zero clicks in 40–49 rows**. L1 keeps them because a dummy that perfectly separates a
handful of rows earns a large coefficient; that is memorisation, not signal.

## 7. Models

| Model | Family | Purpose |
|---|---|---|
| Always-none | floor | Establishes that 98% accuracy is worthless |
| Prior-history rule | heuristic | The benchmark every learned model must beat |
| Logistic regression | linear | Probabilistic baseline |
| Random forest | bagged trees | Non-linear baseline |
| Hist gradient boosting | boosted trees | Strong tabular baseline |
| RBF SVM | margin-based | Distinct geometry |
| MLP (32-16) | neural | Capacity stress test |
| k-NN | instance-based | No fitted boundary |

## 8. Results

### 8.1 Threshold-free, repeated stratified 5-fold ×10

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| prior-history rule | **0.390 ± 0.166** | **0.913 ± 0.049** | 0.092 |
| logistic (prior-clicks only) | 0.325 ± 0.158 | 0.904 ± 0.054 | 0.111 |
| random forest | 0.265 ± 0.162 | 0.874 ± 0.084 | 0.032 |
| logistic regression | 0.258 ± 0.150 | 0.822 ± 0.099 | 0.095 |
| hist gradient boosting | 0.198 ± 0.124 | 0.854 ± 0.094 | 0.022 |
| RBF SVM | 0.166 ± 0.101 | 0.790 ± 0.123 | 0.019 |
| MLP (32-16) | 0.156 ± 0.124 | 0.629 ± 0.176 | 0.085 |
| k-NN (k=15) | 0.139 ± 0.069 | 0.816 ± 0.099 | 0.019 |
| always none | 0.020 ± 0.002 | 0.500 ± 0.000 | 0.020 |

### 8.2 At an 8.5% flag rate

| Model | Recall | Precision | F1 | Achieved flag rate |
|---|---|---|---|---|
| k-NN (k=15) | 0.760 | 0.102 | 0.179 | **15.0%** |
| random forest | 0.708 | 0.170 | 0.272 | 8.6% |
| prior-history rule | 0.702 | 0.172 | 0.273 | 8.3% |
| logistic regression | 0.578 | 0.141 | 0.225 | 8.3% |
| hist gradient boosting | 0.572 | 0.123 | 0.201 | 9.5% |
| RBF SVM | 0.480 | 0.108 | 0.175 | 9.0% |
| MLP (32-16) | 0.343 | 0.083 | 0.132 | 8.6% |

k-NN's recall lead is an artefact: distance-weighted scores are heavily tied, so the
threshold cannot land inside the 8.5% budget and it flags 15% instead. It is not a valid
operating-point comparison, and precision (0.102 vs 0.172) shows why.

### 8.3 Paired per-fold comparison against the rule

| Model | Mean Δ PR-AUC | Std Δ | Folds won (of 50) |
|---|---|---|---|
| random forest | −0.125 | 0.148 | 11 |
| logistic regression | −0.132 | 0.204 | 11 |
| RBF SVM | −0.224 | 0.190 | 4 |
| MLP (32-16) | −0.234 | 0.204 | 7 |
| k-NN (k=15) | −0.251 | 0.155 | 2 |

If the models were equivalent you would expect ~25 wins. They lose on 39–48 of 50 folds,
so this is not variance.

### 8.4 Nested CV (tuning on inner folds, scoring on outer folds)

| Model | Tuned PR-AUC | Default | Gain | Folds better | Reference (same folds) |
|---|---|---|---|---|---|
| logistic regression | 0.310 ± 0.125 | 0.243 | +0.067 | 6/10 | 0.386 |
| RBF SVM | 0.298 ± 0.175 | 0.117 | +0.181 | 10/10 | 0.386 |
| random forest | 0.226 ± 0.109 | 0.182 | +0.044 | 9/10 | 0.386 |
| k-NN | 0.224 ± 0.191 | 0.047 | +0.177 | 10/10 | 0.386 |
| MLP (32-16) | 0.162 ± 0.153 | 0.040 | +0.122 | 7/10 | 0.386 |

Tuning gains track how poor the default was, not how good the model is — the two largest
gains belong to the two worst defaults. Twelve of nineteen parameter-value rows were chosen
in under 80% of folds, so most "tuned" settings are coin flips. The consistent winners
(`C=0.01`, `gamma=0.01`, `min_samples_leaf=5`) all point to heavy regularisation.

## 9. Open-set detection

Fit on known classes only, then score a held-out category against held-out known rows:

| Score | AUROC (6 runs) | Detection at 5% FPR |
|---|---|---|
| **Mahalanobis** | **0.864 – 0.890** | 20–27% |
| max probability | 0.572 – 0.645 | 3–13% |
| k-NN distance | 0.559 – 0.607 | 5–12% |
| decision margin | 0.336 – 0.428 | 0–2% |
| energy | 0.336 – 0.428 | 0–2% |

Three findings:

1. **Feature-space distance works; the classifier head does not.** Mahalanobis nearly
   doubles the AUROC of 1 − max probability.
2. **Energy and margin are numerically identical** in the binary case, because energy is a
   monotone transform of the margin. Energy-based detection earns its reputation in
   multiclass problems, not here.
3. **Head scores point the wrong way** (AUROC < 0.5): unknown rows look *more* clickable
   than known ones.

**Caveat.** The strongest component of the Mahalanobis signal is almost certainly the
one-hot category block going all-zero for an unseen category. That is legitimate for this
data type — "nothing matches a known category" is real evidence — but it will not
generalise to purely continuous features.

## 10. Overfitting and underfitting

| Train rows | LR train AP | LR CV AP | Gap |
|---|---|---|---|
| 20 | 1.000 | 0.121 | 0.879 |
| 60 | 1.000 | 0.142 | 0.858 |
| 100 | 0.961 | 0.286 | 0.675 |

Both tree and linear models memorise the training rows and generalise poorly; the gap
narrows only slowly and the CV curve is still climbing at 100 rows. This is a property of
the dataset, not a hyperparameter mistake — the fix is more data. The MLP is the
underfitting case: low on both sides, and only 7/10 folds improved by tuning.

## 11. Calibration

`class_weight="balanced"` reweights the training distribution, so predicted probabilities
stop estimating the true 2% base rate. The always-none floor has the best Brier score
(0.020) of any model; the balanced logistic regression reaches 0.095 while ranking better
than the floor. If probabilities are to be consumed as probabilities, calibration
(isotonic or Platt on a validation split) is required, or class weights should be dropped
and the threshold tuned instead.

## 12. Error analysis

At the chosen operating point (8.5% flag rate, precision 0.17) roughly **five of every six
flagged impressions do not convert**. That is the expected shape when the base rate is 2%
and precision is 17%: the model's value is a ~8.5× lift over random selection, not a
decision-grade classifier. Recall of ~0.70 means about three in ten clicks are missed.

Failure modes observed:

- All models thin out at the extreme low-probability end; the rare positive with high
  prior clicks (above 1.31) is unreachable by any threshold on that feature.
- k-NN's tied scores make the operating point unrepresentative of the budget.
- Probabilities from weighted models are systematically inflated.

## 13. Threats to validity

1. **Sample size.** 19 positives; a single click moves recall by 5 percentage points, and
   recall confidence intervals are roughly ±20pp. Anything under a 10pp difference is
   unresolved.
2. **One dataset, one seed.** All results come from a synthetic generator that appears to
   encode a near-deterministic relationship in one feature. Real click data is far less
   obliging, and the dominance of `previous_clicks` should be treated as a property of
   this file.
3. **Protocol differences.** Headline tables use repeated stratified 5×10; nested CV uses
   10 outer folds. Numbers from different tables are not directly comparable, and are
   labelled accordingly.
4. **Open-set realism.** The held-out-category protocol is legitimate but underpowered
   for a headline claim, and depends on the categorical encoding described in §9.

## 14. Conclusions

- The deployable artefact is the pipeline, but the **best model is the rule**. A system
  that ignores `previous_clicks ≤ 0.5` in favour of a trained classifier would lose
  accuracy.
- **Extra capacity did not help.** The two most complex models were the two worst, which
  matches the tabular literature and is expected at this sample size.
- **The None problem is solvable in the detection direction.** Mahalanobis distance gives
  a usable open-set signal; softmax confidence does not.
- **Persistence and verification are part of the result.** Every number in this report is
  regenerable from `python main.py`, and the run directory records the data hash, seeds,
  versions and commit that produced it.

## Appendix: artifact map

| Artifact | Contents |
|---|---|
| `manifest.json` | commit, versions, config, data sha256 |
| `cv_folds.csv` / `cv_summary.csv` | baseline CV, per fold and aggregated |
| `advanced_folds.csv` / `advanced_*.csv` | SVM/MLP/k-NN comparison |
| `paired_vs_rule.csv` | per-fold deltas against the rule |
| `nested_cv_*.csv`, `tuning_stability.csv` | tuning results and parameter stability |
| `learning_curve_*.csv` | over/underfitting diagnosis |
| `open_set_results.csv`, `open_set_reports.json`, `none_detectors.csv` | open-set evaluation |
| `feature_comparison.csv`, `feature_stability.csv` | feature engineering and selection |
| `final_selection.json` | the selection decision and its reason |
| `scoring.json`, `baseline_logistic.joblib` | deployment artefacts for `src/predict.py` |
