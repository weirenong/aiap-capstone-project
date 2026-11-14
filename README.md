# aiap-capstone-project

Predicting whether a user clicks an ad, framed as a **"None" classification problem**: the model has to be able to say *"none of my known classes"* instead of always picking one. The target is `clicked_ad` — 19 positives in 955 rows after cleaning, a 2% positive rate — so accuracy is meaningless and every reported number is PR-AUC, recall at a fixed flag rate, or an explicit open-set test.

## Repository map

| Path | What it does |
|---|---|
| `data/synthetic_data_243.csv` | raw input, 1,000 rows, 8 columns |
| `eda.ipynb` | Sprint 2 exploratory analysis, executed with outputs and plots |
| `src/preprocessing.py` | schema, row filtering, impute → engineer → clip → scale, fitted per fold |
| `src/feature_engineering.py` | derived features, in-fold feature selection, selection-stability diagnostics |
| `src/models.py` | baselines, imbalanced-classification metrics, CV comparison, open-set report |
| `src/artifacts.py` | persistence: run directories, manifests, CSV/JSON/joblib round-trip |
| `main.py` | end-to-end validation harness; writes a complete results run |
| `results/` | one directory per run plus `latest.json` pointer |
| `README.md` | this document, with the sprint write-ups below |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py          # validates the pipeline end to end and writes results/run_<timestamp>/
```

`main.py` is a harness, not a demo: every section ends in a `check()` that raises on failure, so a broken assumption exits non-zero instead of printing a plausible-looking table.

## Results and artifacts

Every run writes a self-contained directory, so results survive the process that produced them and later experiments can be compared by reading files instead of re-running:

```
results/
  latest.json                     # newest run id + headline metrics
  run_20261114_022825/
    manifest.json                 # git commit, package versions, data sha256, seeds, config
    index.md                      # human-readable summary of the run
    data_quality.json             # filter report, split sizes, positive rates
    cv_folds.csv / .json          # every fold x model x operating point
    cv_summary.csv / .json        # mean/std across the 50 fits
    baseline_threshold_free.csv   # PR-AUC / ROC-AUC / Brier per model
    baseline_operating.csv        # recall / precision / F1 at the chosen flag rate
    feature_comparison.csv        # feature sets, ranked
    feature_stability.csv         # how often each feature survived selection
    open_set_results.csv          # per held-out category
    open_set_reports.json         # full report objects, one per holdout
    baseline_logistic.joblib      # fitted preprocessor + classifier
```

CSV for tabular results, JSON for structured or nested ones, joblib for the fitted pipeline. Both formats are written for every table because a human reads the CSV and a later script reads the JSON.

Loading is a two-line operation:

```python
from src.artifacts import load_latest_run, load_frame, load_model, compare_runs

run = load_latest_run()
summary = load_frame(run, "cv_summary")
model = load_model(run, "baseline_logistic")
diff = compare_runs(run, other_run, metric="pr_auc_mean")   # matched on model + flag rate
```

Persistence is verified rather than assumed. `main.py` section 11 reloads every artifact it just wrote and checks it: row counts survive the round trip, the reloaded model returns identical probabilities to the original, `latest.json` resolves to the run that was just created, and `compare_runs(run, run)` returns all-zero deltas. An artifact that cannot be loaded back is not persistence, and this is where that gets caught.

---

## Sprint 1: Literature Review — "None" classification

### What the problem actually is

A "None" classification problem is one where the model has to be able to say **"this input is not any of my known classes"**. The same problem appears under several names depending on the field: none-of-the-above, background class, unknown or out-of-scope class, out-of-distribution (OOD) detection, or open-set recognition (OSR). The names aren't interchangeable, and being clear about which one we mean matters, because the three usual settings ask for different things:

- **An explicit none/background class.** The training data contains real none examples and the model learns it as an ordinary label. Cheapest, and only as good as the none data.
- **Reject-only detection.** A closed-set model plus a score that says "probably not one of mine". The model rejects, but doesn't say what the input is.
- **Full open-set recognition.** Reject the unknown and keep classifying the knowns well. This is what the harder benchmarks measure, and it's where the accuracy/rejection trade-off actually hurts.

Standard classifiers assume a closed world. Every input belongs to one of the labels, softmax produces a distribution over those labels, and argmax always returns a winner. The standard survey of the area (Geng et al., 2021) frames the gap as **open space risk**: the part of the feature space far from any known training example, where the model has no evidence but will still confidently commit to something. Recent work also talks about how "open" a problem is, as a function of how many classes are known at training time versus how many appear at test time, so that the difficulty can be stated rather than guessed.

Where this shows up in practice: the "O" tag in named entity recognition, where most tokens are non-entities; out-of-scope intent detection in chatbots; fraud and medical screening; wake-word detection; watchlist face recognition. In all of these the none case is common and the failure mode is expensive.

### Why standard classifiers fail here

Three reasons, and they compound.

**The none class has no coherent structure.** A "cat" class has shared features to learn. The none class is everything else in the universe, so there's no common pattern to fit, only a boundary to draw. Any model that learns it as an ordinary class is learning a very odd, extremely heterogeneous class.

**Severe imbalance.** In fraud, screening, or NER, none examples can outnumber positives by orders of magnitude. Two things follow: a model that always predicts none scores very high on accuracy, and the loss is dominated by none examples unless it's reweighted.

**Softmax can't say "none".** The softmax layer normalises over the known classes so the probabilities sum to one, and the model has no mechanism for putting mass outside the label set. Feed it something unfamiliar and it returns whichever known class looks closest, usually with high confidence. Bendale & Boult (2016) make exactly this argument: the closed-set nature of the network forces a choice among known classes, so a model can be confidently wrong on inputs a human would never label at all.

### Approaches in the literature

**1. Train an explicit none/background class.** The direct fix: add a none class and supply examples. In NER this is just the "O" tag, and it works because the tag comes from the annotation scheme rather than being learned. Elsewhere the quality depends entirely on how representative the negatives are — a none class trained on one kind of negative won't cover another. Hendrycks et al. (2019) push this further with **outlier exposure**: train against an auxiliary set of outliers with a loss that pushes confidence down on them, and detection of *unseen* outliers improves. It's the cheapest method that uses real data, and the first thing I'd try if we have usable negatives.

**2. Threshold a post-hoc score.** No retraining, just a score on top of a closed-set model:
- maximum softmax probability (Hendrycks & Gimpel, 2017) — the baseline everything is measured against, and still surprisingly hard to beat;
- ODIN-style temperature scaling with a small input perturbation, or simply the largest raw logit, which avoids the normalisation problem altogether;
- energy score (Liu et al., 2020) — a free-energy transform of the logits that behaves better than softmax probability and needs no tuning;
- Mahalanobis distance to class-conditional Gaussians in feature space, or distance to the nearest training embedding.

All of these reduce the problem to a one-dimensional threshold, which is a big practical simplification and also the source of most of the evaluation confusion later.

**3. OpenMax and extreme value theory.** Bendale & Boult (2016) take the activation vector from the penultimate layer, fit a Weibull distribution to each class's training activations, and use that to move some probability mass onto an "unknown" pseudo-class at inference. The statistical reasoning is sound, since you're estimating a tail from finite samples, but OpenMax is fiddly to implement and its advantage over a well-tuned threshold is smaller than the framing suggests.

**4. Generative and reconstruction methods.** Fit an autoencoder or GAN and treat high reconstruction error as unknown, or synthesise near-miss unknowns and add them to training. Attractive for multiclass OSR, but they add a second model to validate, and reconstruction error often measures "is this pattern unusual" rather than "is this class unknown".

**5. Placeholder and prototype heads.** Rather than one none class, learn placeholder prototypes that represent "otherness" in the embedding space, with a margin keeping known classes away from them (Zhou et al., 2021). A middle ground between an explicit none class and a pure post-hoc score, and usually better known-class accuracy at a given rejection rate.

**6. One-class fallbacks.** With no usable negatives at all, the problem becomes unsupervised: fit a boundary around the known data (one-class SVM, SVDD) or use an isolation-based anomaly score. Worth keeping as a baseline, not as the main method, since it throws away the known-class labels.

**7. The reality check.** Vaze et al. (2022) ran a controlled comparison and found OSR difficulty is driven mainly by **semantic shift** — how far the unknown classes sit from the known ones — and that a well-trained closed-set classifier with a simple max-logit or softmax-confidence score matches or beats many dedicated OSR methods. Their conclusion is a useful corrective: build the strongest closed-set model you can, then treat none-handling as a score on top of it.

### Data and feature engineering for the none class

A few things specific to this problem that don't appear in ordinary classification write-ups:

- **Negatives decide everything.** The none examples in training define the boundary. If deployment will contain a different flavour of unknown, it has to show up in validation, or the reported number is meaningless.
- **Split at the class level, not the sample level.** The biggest protocol mistake in this area. If unknown-class rows are mixed into training as ordinary negatives, the model has effectively seen the unknown class and the evaluation is invalid. The standard protocol holds whole classes out and evaluates on those.
- **Don't SMOTE the none class.** Synthetic minority interpolation (Chawla et al., 2002) assumes the minority class has local structure worth interpolating within. The none class is unbounded and heterogeneous, so interpolating between two unrelated unknowns produces a point that is neither. Here over-sampling is more likely to distort the boundary than help it — class weights or a term that penalises confident none predictions are the better lever.
- **Margin and reweighting help indirectly.** Focal loss, class weights, or a margin-based embedding loss all improve separation between knowns, which makes the rejection score cleaner. Often a bigger win than changing the score function.
- **Normalise the feature space.** Most distance-based scores assume an embedding where Euclidean distance means something, so L2 normalisation and a sensible choice of layer matter more than they should.
- **Tune the threshold on validation**, and per class if some known classes sit naturally closer to the unknowns than others.
- **For sequence labelling**, the none class is a tag and the real problem is boundaries. Structured output on top of the encoder helps because the none tag's probability depends on its neighbours, and token accuracy is dominated by the "O" tag, so entity-level F1 is the honest metric.

### Evaluation

Accuracy is worse than useless here, because it rewards the always-none model, which is the exact failure mode we're trying to avoid.

What's reported in this literature:

- **AUROC** of the none score, knowns as negatives and unknowns as positives. Threshold independent and comparable across methods, so it's the default headline number.
- **AUPR** — more informative when the unknown rate is low, which is the realistic deployment case.
- **FPR at 95% TPR** — the false positive rate when the threshold catches 95% of unknowns. This is the metric that corresponds to an operating point, and the OpenOOD benchmark (Yang et al., 2022) standardises it precisely because AUROC hides whether a method is usable.
- **Open-set classification rate / normalised accuracy** for full OSR: known-class accuracy and unknown rejection in one number, reported alongside plain closed-set accuracy so the cost of handling none is visible.
- **Detection error** — the minimum error over thresholds, useful when you don't want to commit to one.
- **Calibration** (ECE, reliability plots) if the score will be shown to a human or drive a decision. A confidence of 0.9 meaning nothing is the normal state, not a bug.
- **The reject-option view** (Chow, 1970) is worth knowing: there's a fundamental error-reject trade-off, so any single number without a stated operating point is incomplete.

Two pitfalls. AUPR depends heavily on which unknown set was used and whether it's near or far from the knowns, so numbers from different papers aren't comparable unless the protocol matches. And protocol differences are big enough that OpenOOD exists to fix them — averaging over a mix of near and far unknowns can flatter a method that only works on obvious cases.

### Validation methodology

- Split by class: known classes in train, unknown classes held out entirely, and only those held-out classes used for the none evaluation.
- Cross-validation has to respect that split, or the same unknown class shows up in train in one fold and test in another.
- Run more than one unknown set. A method that holds up on far-OOD may collapse on near-OOD, and that spread is the result, not a nuisance.
- Choose the threshold on training/validation data only. Picking it on the test unknowns is leakage and inflates everything.
- Fix seeds and pin versions. The score distributions are close enough between methods that run-to-run variance can be mistaken for an improvement.

### What this means for our approach

Build the strongest closed-set classifier first, since that's most of the battle. Then add a none score: max softmax probability as the baseline that has to be beaten, energy or Mahalanobis if it isn't, and an explicit none class with outlier exposure only if we can get representative negatives. Pick the threshold on validation at a stated operating point (probably 95% TPR), report the whole curve next to it, and report closed-set accuracy, AUROC, AUPR and open-set accuracy at that threshold across at least two unknown sets. State the exact known/unknown split — without it the numbers mean nothing to whoever reads this later.

### Limitations

A short review, not a survey. Two gaps worth naming: I haven't covered open-set work for text beyond the sequence-labelling case, and the tabular side is thin in the literature generally, so most of the benchmark evidence here comes from vision. Only ten sources are cited; the field is much larger, and the ones chosen are the ones a first implementation would actually touch. Bibliographic details were checked against publisher and arXiv records, but I haven't quoted specific numbers from any of these papers, so nothing here depends on a figure I didn't reproduce.

### References

Bendale, A., & Boult, T. E. (2016). Towards open set deep networks. *CVPR*. doi:10.1109/CVPR.2016.173

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16. doi:10.1613/jair.953

Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1). doi:10.1109/TIT.1970.1054406

Geng, C., Huang, S.-J., & Chen, S. (2021). Recent advances in open set recognition: a survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43(10). doi:10.1109/TPAMI.2020.2981604

Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR*. arXiv:1610.02136

Hendrycks, D., Mazeika, M., & Dietterich, T. (2019). Deep anomaly detection with outlier exposure. *ICLR*. arXiv:1812.04606

Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. *NeurIPS*. arXiv:2010.03759

Vaze, S., Han, K., Vedaldi, A., & Zisserman, A. (2022). Open-set recognition: a good closed-set classifier is all you need? *ICLR*. arXiv:2110.06207

Yang, J., Wang, P., Zou, D., Zhou, Z., Ding, K., Peng, W., Wang, H., Chen, G., Li, B., Sun, Y., Du, X., Zhou, K., Zhang, W., Hendrycks, D., Li, Y., & Liu, Z. (2022). OpenOOD: benchmarking generalized out-of-distribution detection. *NeurIPS Datasets and Benchmarks*. arXiv:2210.07242

Zhou, D.-W., Ye, H.-J., & Zhan, D.-C. (2021). Learning placeholders for open-set recognition. *CVPR*. doi:10.1109/CVPR.2021.00438

---

## Sprint 2: Exploratory Data Analysis

Full analysis and plots are in `eda.ipynb`; this is the summary of what it found and what it changes.

**Data.** `data/synthetic_data_243.csv`, 1,000 rows, 8 columns. Target is `clicked_ad` (binary). Features: `user_age` and `previous_clicks_on_similar_ads` (numeric), plus `gender`, `device_type`, `time_of_day`, `ad_placement_position`, `ad_category` (categorical).

### Data quality

The file is unusually clean for a project dataset: **no missing values, no duplicate rows, and all 1,000 rows have a unique feature combination.** So imputation and de-duplication aren't workstreams at all, and there's no risk of a dedup step silently changing the split.

Two things do need a decision:

- **45 rows have `user_age` under 18**, minimum 6.99. For an ad-click problem that's a legal and ethical issue more than a modelling one, and it also puts a tail on the age distribution that isn't the target population. Filter it or document it, but decide before splitting.
- **`previous_clicks_on_similar_ads` is 99.4% fractional**, so despite the name it isn't a count in this file — it's a continuous variable. Treat it as continuous and don't build "has never clicked" style features without agreeing where the cut goes.

### The target

20 positives out of 1,000 — a 2.0% click rate, a 1:49 imbalance, and 98% accuracy for a model that predicts "none" every time (95% Wilson interval on the rate: 1.3%–3.1%).

Two consequences. Accuracy is useless as a headline metric, which was already the assumption from Sprint 1. And **20 positives is small enough that one misclassified click moves recall by 5 percentage points**, so a recall estimate carries a 95% interval around ±20pp. Differences under roughly 10pp between models are not meaningful on this data, and that constraint should be stated in the results rather than papered over.

### What actually predicts a click

`previous_clicks_on_similar_ads` is the whole story. Clickers average 0.45 prior clicks, non-clickers 1.96 (Welch p ≈ 0, Mann-Whitney p ≈ 0), and binned, the click rate is 15.8% at or below 0.5 prior clicks and **0 out of 391 rows above 2**. The direction is counter-intuitive — more prior clicks means less likely to click — which in synthetic data probably reflects how the generator was parameterised rather than user behaviour. Worth saying out loud instead of interpreting it.

`ad_category` is the only other feature with a plausible effect (chi² = 22.76, p = 0.0004): Gaming 7.0%, Travel and Education 4.0%, Technology 0.8%, E-commerce and Finance 0.7%. Gender is borderline (p = 0.12) but that's 6 clicks among 480 women and 0 among 40 in the "Other" level, so it's noise as much as signal. Device type, time of day and placement position show nothing (p > 0.5). `user_age` is flat (p = 0.63).

Mutual information agrees on the top feature and then falls apart — Gaming scores 0.000 despite having the highest click rate, and `time_of_day_Evening` ranks second despite a non-significant chi-square. With 20 positives those tail rankings are unstable across seeds, so MI is confirmation that one feature dominates, not a feature-selection ranking.

### The baseline to beat

A single threshold, `previous_clicks_on_similar_ads <= 0.5`, gives **recall 0.80 at precision 0.16** — 85 false positives, an 8.5% flag rate against a 2% base rate. Pushing to `<= 1.31` catches all 20 clicks but flags 390 rows (precision 0.05). Any model that can't beat 0.80 recall at an 8.5% flag rate has added nothing, so the single-rule result is the thing every later model gets compared to.

### What this means for the "None" problem

Two readings, and they need different evaluation:

- **As built**, the none class is the 98% background: "this user clicks none of the ads". The project is entirely about the minority class and about keeping the false-positive rate usable, which is the framing the rest of the work should use.
- **As an open-set problem**, `ad_category` is the natural class space: train on five categories, hold one out, and check whether clicks on the held-out category get flagged as none or confidently misclassified. This is the honest open-set test for tabular data, but it's underpowered here — holding out Travel leaves 6 positives, holding out Gaming removes 7 of the 20. Feasible as a sanity check, not as a headline result.

### Preprocessing decisions

- No imputation pipeline needed; no dedup step.
- One-hot encoding for all categoricals — max cardinality is 6 (`ad_category`), so no target encoding and no encoding leakage risk.
- Scaling only matters for logistic regression and k-NN, and it goes inside the pipeline either way.
- Class weights first, threshold tuning second, resampling only as a comparison and only inside the fold.
- Metrics: PR-AUC and recall at a fixed flag rate (8.5% or 10%) as the primary pair, accuracy and ROC-AUC as secondary, and the spread across folds reported instead of a bare number.
- Validation: repeated stratified 5-fold, because 10-fold leaves about 2 positives per fold and 5-fold about 4 — either is thin, so repeated 5-fold with a reported spread is the least misleading option.
- Leakage risk: nothing in the columns is a proxy for the label and nothing is computed after the click, so the realistic risk is only resampling or feature selection outside the split.

### Limitations of this EDA

One dataset of 1,000 rows and 20 positives. Every p-value above is indicative at best — several expected cell counts in the chi-square are around 1 — and the correlations are dominated by a single-lever synthetic generator. The open-set variant of the problem can't be evaluated credibly at this sample size.

---

## Sprint 3: Preprocessing Pipeline

Code: `src/preprocessing.py` (the module), `main.py` (validation harness, `python main.py`). The EDA notebook is unchanged; everything below is reproducible from the two files.

### Design rules, and why

The pipeline is built around the fact that the none class is 98% of the data, so the usual preprocessing defaults actively work against the problem.

1. **Row filtering happens before the split, never inside a Pipeline.** Dropping rows inside a `Pipeline` desynchronises `X` and `y`, so eligibility rules live in `filter_eligible_rows()` and are applied once, up front.
2. **Nothing that learns from data is fitted before the split.** Imputation values, IQR fences, scaler statistics and the category vocabulary all come from `fit` on the training fold. `main.py` asserts this rather than trusting it: the stored fences and scaler means are compared against train-only values, and against full-data values to confirm they would have differed.
3. **No resampling in the preprocessing chain.** SMOTE belongs inside the CV loop, so `build_resampled_model()` wraps it in an imbalanced-learn pipeline. Having it as a separate function means it can't be called early by accident, which is the mistake the imbalanced-learn docs warn about.
4. **Unseen categories encode to an all-zero block** (`handle_unknown="ignore"`). For a None problem that's the correct behaviour: an unseen category is evidence against every known class, not a reason to crash or guess. `main.py` injects a category that doesn't exist and checks the block is zeros.
5. **The minority class is never dropped, capped or reordered.** No step removes positives.

### Pipeline structure

```
numeric (2 cols):   PriorHistoryFeatures -> SimpleImputer(median) -> IQRClipper(1.5) -> StandardScaler
categorical (5):    SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore")
```

24 features out of 7 in: 4 numeric (the two originals plus two derived) and 20 one-hot flags. Clipping comes after imputation so NaNs don't poison the quantiles, and scaling comes last so the scaler sees the values the model will actually get.

`IQRClipper` clips rather than drops. With 20 positives, deleting outlier rows is not an option, and the tail beyond the fence carries little information at this sample size.

### Feature engineering

The EDA showed the effect of `previous_clicks_on_similar_ads` is a step, not a line: 15.8% click rate at or below 0.5, zero clicks above 2. A linear model can't represent a step, so `PriorHistoryFeatures` adds two stateless derived columns:

- `previous_clicks_on_similar_ads__low_history` — `1.0` when prior clicks ≤ 0.5.
- `previous_clicks_on_similar_ads__log1p` — to soften the right skew for the linear model.

That flag is what closes most of the gap: logistic regression on the raw features reaches recall 0.50 at an 8.5% flag rate, and 0.75 with the flag added. Trees were already at 0.75 without it, which is consistent with the EDA explanation — they can learn the step on their own.

Two honest caveats. The 0.5 cut was chosen while looking at the whole dataset during EDA, so it carries a little optimism even though it's a fixed constant rather than a fitted parameter; Sprint 4 should confirm inside CV that the flag still earns its place. And the flag is only useful because the dataset is synthetic and monotone — real click data would be less obliging.

### A bug worth recording

The first working version had the low-history flag as a column of zeros. `IQRClipper` computed IQR fences for it, and because only ~9% of rows are 1, the 25th and 75th percentiles are both 0 — so the fences were 0 and 0, and clipping silently erased the feature. Logistic regression then scored 0.50 recall while the same model outside the pipeline scored 0.75.

Fixed by leaving zero-IQR columns unclipped: a binary flag or a constant has no tail to trim. The regression is now pinned by a check in `main.py` that the flag survives preprocessing with two distinct values and stays aligned with the raw rule. Worth keeping in mind for any future derived column: a clipping step applied to the whole numeric block will quietly destroy anything without spread.

### What the harness checks

Sections 1–8 of `main.py`, all passing:

| Check | Result |
|---|---|
| Under-18 filter removes rows | 45 of 1000, and **1 of 20 positives (5% of the minority class)** |
| Duplicates | none, as the EDA found |
| Stratified split | 764 train / 15 positives, 191 test / 4 positives |
| No NaN after transform | pass, train and test |
| Feature count matches schema | 24 == 24 |
| Low-history flag survives and stays aligned | pass (after the fix above) |
| IQR fence fitted on train, not full data | pass, per column |
| Scaler mean fitted on clipped train | pass, per column |
| Unseen category → all-zero block | pass |
| Determinism across two fits | identical output |

### End-to-end result (smoke test, not a result)

Logistic regression through the pipeline, threshold picked on train at an 8.5% flag rate, evaluated on the 191-row test split:

- **TP 3, FP 16, FN 1 → recall 0.75, precision 0.16, flag rate 9.9%.**
- EDA single-rule baseline: recall 0.80, precision 0.16 at 8.5%.

So the pipeline is competitive with the rule but does not beat it, on a test set with four positives. That is not a result to report — it's evidence the plumbing is correct and a warning that the modelling sprint has to be careful about celebrating differences that are one row wide.

### The open-set behaviour is still absent

`make_open_set_split()` fits on five ad categories and scores the held-out one. Held-out rows score **higher** (mean predicted click probability 0.211) than known test rows (0.160), so the model does not treat them as none-of-the-above at all — if anything it considers them more clickable. That is the expected failure from Sprint 1: a closed-set classifier has no mechanism for saying "not mine", and the extra positive-class weighting pushes it the wrong way. Sprint 4 needs an actual none score (max softmax, energy, or distance) before this question can be asked properly.

### Decisions left open

- **The under-18 filter costs 5% of the minority class.** It's the right call ethically, but it is a real cost at 20 positives — confirm it's staying before Sprint 4, and report 19 positives rather than 20 from here on.
- **The 0.5 low-history cut** is EDA-derived. Re-check it inside CV rather than treating it as fixed.
- **`remove_outliers` is deliberately not implemented as row deletion.** If the modelling sprint wants to compare clipping against dropping, that's a separate branch, not a flag on this pipeline.

---

## Sprint 3, Issue 2: Feature Engineering and Selection

Code: `src/feature_engineering.py`. Run the comparison with `python main.py` (section 9). `preprocessing.py` now imports its `PriorHistoryFeatures` from here, so there is one definition and a one-way dependency.

### What's in the module

**Feature creation** — three transformers, all `fit`-on-the-fold only:

| Builder | What it adds | Why |
|---|---|---|
| `PriorHistoryFeatures` | `prev_clicks__log1p`, `prev_clicks__low_history` | The EDA showed a step at ≤ 0.5 prior clicks, not a slope; a linear model can't represent a step. `log1p` softens the skew. |
| `AgeBandFeatures` | 5 one-hot age bands | Age was flat (p = 0.63), which is a reason to test for a non-linear effect, not a reason to drop the column. |
| `PairwiseInteractionFeatures` | one-hot crosses, e.g. device × time of day | The EDA said any remaining signal would be an interaction, so the hypothesis is testable instead of assumed. Unseen combinations encode to all zeros, same logic as `handle_unknown="ignore"`. |

**Feature selection** — `FeatureSelector`, a transformer that sits *inside* the pipeline so it is refitted per fold. Two strategies: mutual information (top-k) and L1 logistic coefficients. Selection outside the fold is the same leakage as scaling on the full frame, and it is the more common version of it.

**Diagnostics** — `evaluate_pipelines` (repeated stratified 5×10 CV, PR-AUC and recall at a fixed flag rate, threshold taken from the training fold each time) and `selection_frequency` (how often each feature survives across 50 fits). `selection_frequency` is the point: at 19 positives a single global ranking is noise, so the honest question is whether a feature survives refits.

### Results

Repeated stratified 5-fold ×10 on all 19 positives. Ranked by PR-AUC; the std column is the one that matters:

| Feature set | n | PR-AUC mean | PR-AUC std | Recall mean | Recall std |
|---|---|---|---|---|---|
| + prior history + select (MI) | 12 | 0.2705 | 0.1372 | 0.622 | 0.197 |
| + prior history + age bands | 29 | 0.2641 | 0.1614 | 0.582 | 0.211 |
| + prior history | 24 | 0.2581 | 0.1481 | 0.593 | 0.220 |
| + prior history + interactions | 53 | 0.2424 | 0.1601 | 0.552 | 0.231 |
| + prior history + select (L1) | 12 | 0.2408 | 0.1329 | 0.597 | 0.214 |
| raw (no derived) | 22 | 0.2299 | 0.1499 | 0.490 | 0.249 |

Reading it honestly:

- **The prior-history features are the only clear win.** PR-AUC 0.230 → 0.258 and recall 0.490 → 0.593, and it is the one change with a mechanism behind it. Everything else is inside a standard deviation.
- **Age bands add nothing** (0.264 vs 0.258, std 0.16). Four extra columns for a non-effect, which at 19 positives is a real cost.
- **Interactions make it worse** (0.242), and they add 29 columns. The hypothesis is rejected — there isn't enough data for a 12-way or 18-way cross to mean anything.
- **Selection does not help.** MI selection has the best mean but drops 12 columns to get there, and its std is as large as the difference it claims. L1 selection is actively worse than no selection. No feature set here beats the EDA's single-threshold rule (recall 0.80 at an 8.5% flag rate) on any comparison.

### The selection stability result is the interesting part

8 of 24 features survive ≥80% of the 50 fits. The three most stable are:

- `device_type_Tablet` — 49 / 50, with **0 clicks in 49 rows**
- `gender_Other` — 49 / 50, with **0 clicks in 40 rows**
- `ad_placement_position_Footer` — 48 / 50, with **0 clicks in 49 rows**

Those aren't features the model found useful; they're small cells where nobody clicked. L1 keeps them because a dummy that perfectly separates a handful of rows gets a large coefficient, and the "signal" is memorisation of an empty cell. This is exactly the failure mode to expect when selecting from 24 columns with 19 positives, and it's why selection stays a diagnostic here rather than a step in the final pipeline.

### Bugs found, both silent

1. **`PairwiseInteractionFeatures` produced an all-zero block.** `pd.get_dummies` labels its columns with the bare level, but the code reindexed against prefixed names (`device_type__x__time_of_day=Mobile|Morning`), so every column was dropped and refilled with zeros. The comparison then reported "interactions add nothing" when in fact interactions were never there — identical metrics to the no-interaction run to four decimals, which is what gave it away. Fixed by renaming before reindexing, and now pinned by checks that each block is non-degenerate and has exactly one active level per row.
2. **The selector reported `x0..x23` instead of feature names**, because `ColumnTransformer` hands it a numpy array. Fixed by pulling names from the preprocessor step and applying the selector's support mask.

Both are the same class of problem as the clipping bug in Issue 1: a transform that silently produces zeros looks like a null result. The guards in `main.py` exist specifically because "no effect" and "not applied" are indistinguishable in a metrics table.

### Decisions for Sprint 4

- **Use `+ prior history` as the feature set.** Skip age bands and interactions; both are noise at this sample size and cost columns.
- **Keep selection as a diagnostic, not a pipeline step.** The stability table is worth reporting; the selected features are not worth trusting.
- **Stop treating model differences under ~0.10 PR-AUC as real.** That is one standard deviation here.
- **The single-rule baseline still wins.** Any sprint-4 model has to beat recall 0.80 at an 8.5% flag rate, and nothing so far has.
- **The open-set score is still missing.** Feature engineering didn't touch it, and the Sprint 3 smoke test showed held-out categories scoring *higher* than known ones. That is the next real problem, not more tabular features.

---

## Sprint 4, Issue 1: Baselines and Benchmark

Code: `src/models.py`. Run `python main.py` (section 10). Uses the Sprint 3 pipeline and feature set unchanged, so the numbers are comparable with everything before this.

### What's in the module

**Baselines** — chosen to bracket the problem rather than to look impressive:

| Model | What it establishes |
|---|---|
| `AlwaysNoneClassifier` | The floor. 98% accurate, zero recall. Included because the accuracy trap is the whole reason the metrics look the way they do. |
| `PriorHistoryRuleClassifier` | The EDA rule (`previous_clicks ≤ 0.5`). This is the bar, not the floor — it already beat every learned model in Sprint 3. |
| Logistic regression | Linear baseline through the full pipeline, balanced class weights. |
| Logistic, prior-clicks block only | Diagnostic: what happens when the cohort features are removed. |
| Random forest | Non-linear baseline, `balanced_subsample`, shallow leaves. |
| Hist gradient boosting | Strong tabular baseline without class weighting (it has no `class_weight`). |

**Metrics** — threshold-free (`PR-AUC`, `ROC-AUC`, `Brier`) plus operating points (`precision`, `recall`, `F1`, achieved flag rate) at 5%, 8.5% and 10% flag rates, with the threshold always taken from the training fold. Accuracy is deliberately absent from the code path, not just from the write-up.

**Open-set evaluation** — `none_score` (1 − max predicted probability) plus `open_set_report`, which fits on five ad categories and scores a held-out sixth. This is the measurement Sprint 3 flagged as missing.

### Results

Repeated stratified 5-fold ×10, all 19 positives. Threshold-free first:

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| prior-history rule | **0.390 ± 0.166** | **0.913 ± 0.049** |
| logistic (prior-clicks only) | 0.325 ± 0.158 | 0.904 ± 0.054 |
| random forest | 0.265 ± 0.162 | 0.874 ± 0.084 |
| logistic regression | 0.258 ± 0.150 | 0.822 ± 0.099 |
| hist gradient boosting | 0.198 ± 0.124 | 0.854 ± 0.094 |
| always none (floor) | 0.020 ± 0.002 | 0.500 ± 0.000 |

At an 8.5% flag rate, threshold from the training fold:

| Model | Recall | Precision | F1 | Achieved flag rate |
|---|---|---|---|---|
| random forest | **0.708 ± 0.211** | 0.170 | 0.272 | 8.6% |
| prior-history rule | 0.702 ± 0.245 | 0.172 | 0.273 | 8.3% |
| logistic (prior-clicks only) | 0.645 ± 0.214 | 0.160 | 0.253 | 8.3% |
| logistic regression | 0.578 ± 0.219 | 0.141 | 0.225 | 8.3% |
| hist gradient boosting | 0.572 ± 0.221 | 0.123 | 0.201 | 9.5% |
| always none (floor) | 0.000 ± 0.000 | 0.000 | 0.000 | 0.0% |

### What the benchmark actually says

1. **The floor is 98% accurate and useless.** Zero recall at any threshold, because a constant score cannot rank. That is the concrete version of the accuracy argument from the EDA.
2. **A hand-written rule is still the best model on this data.** PR-AUC 0.390 and ROC-AUC 0.913, ahead of every learned model. The random forest matches it at one operating point (0.708 vs 0.702 recall) but needs 400 trees and 24 features to tie a one-line threshold, and ranks worse.
3. **The cohort features cost accuracy rather than adding it.** Logistic regression with the prior-clicks block alone scores 0.325 PR-AUC and 0.645 recall; adding device, time, placement, category and gender drops it to 0.258 and 0.578. At 19 positives those columns are variance, not signal — the same conclusion Sprint 3 reached, now visible in model performance instead of feature selection.
4. **Differences inside the standard deviations are not results.** Recall means span 0.57 to 0.71 with std around 0.21. Only the rule-vs-floor and rule-vs-full-logistic gaps are large enough to defend.
5. **Nothing here is production-ready.** The best operating point finds ~70% of clicks while flagging 8.5% of all impressions, at a precision of 0.17 — for a 2% base rate, that's a lift of about 8×, which is a reasonable start and a long way from a decision system.

### The open-set result

Held-out category as the unknown class, none score = 1 − max probability:

| Held out | n unknown | AUROC | Detection at 5% FPR |
|---|---|---|---|
| Gaming | 94 | 0.642 | 3.2% |
| Travel | 143 | 0.607 | 7.7% |
| Education | 44 | 0.645 | 4.5% |

0.5 is chance. 0.61–0.65 is barely above it, and at a 5% false-positive budget the score catches 3–8% of unknowns — in practice, nothing. Mean none scores are higher for unknown rows than known ones (0.10–0.14 vs 0.07–0.09), so there is a trace of signal, but nowhere near usable. This is the measured form of the gap Sprint 1 opened: a closed-set classifier has no mechanism for saying "none of mine", and dressing it with 1 − max softmax is not one. **Update:** Issue 2 shows this is a limit of the *score*, not the problem — a Mahalanobis detector reaches 0.86–0.89 AUROC on the same holdouts.

The three holdouts are also the honest caveat — semantic shift dominates open-set difficulty, so a single holdout would have been meaningless either way.

### Two bugs found in the metrics code

Both changed conclusions, which is why they're recorded rather than quietly fixed:

1. **The threshold search ran descending**, so it returned the maximum score and flagged almost nothing — the always-none floor appeared to have recall 1.0, and every model showed ≈0 recall at an 8.5% budget. Fixed by walking distinct scores ascending and taking the first that fits the budget.
2. **Ties at a constant score.** A quantile threshold on an all-equal score flags every row, which handed the floor recall 1.0 again. Fixed by treating the budget as a hard cap: if no threshold fits, nothing is flagged.

The lesson is the same one from Sprints 3.1 and 3.2 — a metric can look like a result while the machinery underneath is degenerate. The floor model exists partly to catch this: if "predict none always" ever shows non-zero recall, the evaluation is broken.

### Established benchmark for Sprint 5

- **To beat:** prior-history rule — PR-AUC 0.390, ROC-AUC 0.913, recall 0.702 at an 8.5% flag rate.
- **Realistic target:** a model that beats 0.39 PR-AUC outside its standard deviation, on features that aren't just prior clicks, without resampling leakage.
- **Also required:** an open-set score with AUROC meaningfully above 0.65 and a detection rate that isn't inside a 5% false-positive budget. Nothing in Issue 1 does that; Issue 2 finds a detector that does (Mahalanobis, 0.86–0.89 AUROC).
- **Report the std.** At 19 positives a bare mean is a claim about a coin.

---

## Sprint 4, Issue 2: Persisting Results and Artifacts

The pipeline previously produced clean tables and report objects, then printed them and let them die with the process. That's fixed: `src/artifacts.py` adds a run-directory layer, and `main.py` section 11 now writes ten artifacts per run and verifies every one of them loads back.

### What gets written

| Artifact | Contents |
|---|---|
| `manifest.json` | Run id, UTC timestamp, git commit, Python and package versions, data file sha256 and size, full config (target, features, flag rates, CV settings, seed, cut points) |
| `data_quality.json` | Filter report (45 under-18 rows removed, 1 positive lost), split sizes, positive rates |
| `cv_folds.csv/.json` | All 600 rows: every fold × model × flag rate, with thresholds and confusion counts |
| `cv_summary.csv/.json` | Mean and std per model and flag rate, including Brier |
| `baseline_threshold_free.csv` | PR-AUC, ROC-AUC, Brier per model |
| `baseline_operating.csv` | Recall, precision, F1, achieved flag rate at 8.5% |
| `feature_comparison.csv` | The six feature sets from Sprint 3.2, ranked |
| `feature_stability.csv` | Per-feature survival across 50 fits |
| `open_set_results.csv` | Per held-out category: AUROC, detection at 5% FPR, mean none scores |
| `open_set_reports.json` | Full `OpenSetReport` objects, one per holdout |
| `baseline_logistic.joblib` | Fitted preprocessor + logistic regression, loadable directly |
| `index.md` + `results/latest.json` | Human summary and the pointer to the newest run |

Aggregated and fold-level results are both kept, because the mean alone hides the variance that dominates this dataset. Two formats for each table: CSV for reading, JSON for scripts.

### Why this matters beyond tidiness

- **Comparisons become reads.** `compare_runs(old, new, metric="pr_auc_mean")` joins two persisted `cv_summary` tables on model and flag rate and returns the deltas. Previously this required re-running both experiments.
- **Provenance is explicit.** The manifest records the data file's sha256, so a result can be tied to exact input bytes rather than a filename. If the CSV changes, the hash no longer matches.
- **The baseline is reusable.** `load_model(run, "baseline_logistic")` returns the fitted pipeline — preprocessor included, so it can be applied to new rows without rebuilding anything.
- **Runs don't overwrite each other.** Directories are timestamped, with a collision suffix for two runs in the same second, and `latest.json` points at the most recent.

### Verification

Section 11 fails the run if any of these break:

| Check | Result |
|---|---|
| Manifest written | pass |
| `cv_folds` row count survives the round trip | pass (600 rows) |
| `cv_summary` row count survives | pass (12 rows) |
| Open-set reports reload | pass (3 reports) |
| Reloaded model predicts identically | pass, `np.allclose` to the original |
| `latest.json` resolves to this run | pass |
| `compare_runs(run, run)` returns zero deltas | pass |

None of this is mocked: the checks read the files back off disk through the same public loaders a later script would use.

### Reproducing a run

```bash
pip install -r requirements.txt
python main.py
cat results/latest.json
```

The manifest's `git_commit`, `random_state`, and `data.sha256` are what make a run reproducible. `requirements.txt` pins exact versions; `joblib==1.6.0` was added for the model artifact.

### What persisting the numbers surfaced

Aggregating Brier alongside PR-AUC made a calibration problem visible that the previous run had thrown away:

| Model | PR-AUC | Brier |
|---|---|---|
| always none (floor) | 0.020 | **0.0199** |
| hist gradient boosting | 0.198 | 0.0222 |
| random forest | 0.265 | 0.0317 |
| logistic regression | 0.258 | 0.0951 |
| prior-history rule | 0.390 | 0.0918 |
| logistic (prior-clicks only) | 0.325 | 0.1109 |

The floor has the best Brier score of any model. That isn't a paradox — it emits `[1, 0]` probabilities and is right 98% of the time, which is exactly what Brier rewards when the base rate is 2%. What it exposes is that `class_weight="balanced"` reweights the training distribution, so the model's probabilities no longer estimate the real 2% rate: the balanced logistic regression ranks better than the floor (PR-AUC 0.258 vs 0.020) but is five times worse as a probability estimate. Hist gradient boosting, which has no class weighting, keeps a Brier of 0.022 — close to the floor — while still ranking well.

The practical consequence: if these probabilities are ever going to be used as probabilities (expected-value decisions, ranking by expected revenue, a dashboard), the reweighted models need calibration — isotonic or Platt on a validation split — or the class weights should be dropped and the threshold tuned instead. That was invisible while results were printed and discarded; persisting Brier made it a one-line comparison.

The rule model's Brier (0.092) is the one number in that table that shouldn't be read at all: its `predict_proba` is an arbitrary squashing of a distance, not a calibrated probability, and it's included only so the interface is uniform.

### What changed in the repo

- `src/artifacts.py` — new, the persistence layer.
- `main.py` — section 11, plus the Brier column flowing into the summary.
- `requirements.txt` — `joblib` added, pins grouped by purpose.
- `results/` — now populated by each run; `results/latest.json` points at the newest one.

---

## Sprint 4, Issue 2: Advanced Models and None Detection

Code: `src/advanced_models.py`. Run `python main.py` section 11. Three architectures were added; the baselines are reused verbatim from Issue 1 so the tables describe the same models rather than similarly-named ones.

### The three architectures

| Model | Family | Why it's a fair second opinion |
|---|---|---|
| RBF SVM | margin-based | Different geometry to trees and to a linear model; needs scaling, which the pipeline already does. `probability=True` adds Platt calibration per fold. |
| MLP (32, 16) | neural | The architecture "advanced" usually implies. Included because it should be tested, not because 19 positives can support it. |
| k-NN (k=15, distance-weighted) | instance-based | No fitted decision boundary at all — the natural contrast to everything else, and the classifier whose score ties matter most. |

### Results

Threshold-free, 5-fold ×10:

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| prior-history rule | **0.390 ± 0.166** | **0.913 ± 0.049** | 0.092 |
| random forest | 0.265 ± 0.162 | 0.874 ± 0.084 | 0.032 |
| logistic regression | 0.258 ± 0.150 | 0.822 ± 0.099 | 0.095 |
| **RBF SVM** | 0.166 ± 0.101 | 0.790 ± 0.123 | 0.019 |
| **MLP (32-16)** | 0.156 ± 0.124 | 0.629 ± 0.176 | 0.085 |
| **k-NN (k=15)** | 0.139 ± 0.069 | 0.816 ± 0.099 | 0.019 |

At an 8.5% flag rate:

| Model | Recall | Precision | F1 | Achieved flag rate |
|---|---|---|---|---|
| k-NN (k=15) | 0.760 | 0.102 | 0.179 | **15.0%** |
| random forest | 0.708 | 0.170 | 0.272 | 8.6% |
| prior-history rule | 0.702 | 0.172 | 0.273 | 8.3% |
| logistic regression | 0.578 | 0.141 | 0.225 | 8.3% |
| RBF SVM | 0.480 | 0.108 | 0.175 | 9.0% |
| MLP (32-16) | 0.343 | 0.083 | 0.132 | 8.6% |

### Relative performance, paired

Same folds for every model, so the useful statistic is the per-fold difference from the reference rather than two independent means:

| Model | Mean Δ PR-AUC | Std Δ | Folds won (of 50) |
|---|---|---|---|
| random forest | −0.125 | 0.148 | 11 |
| logistic regression | −0.132 | 0.204 | 11 |
| RBF SVM | −0.224 | 0.190 | 4 |
| MLP (32-16) | −0.234 | 0.204 | 7 |
| k-NN (k=15) | −0.251 | 0.155 | 2 |

Every model loses to a one-line threshold, and loses on 39–48 of 50 folds. This isn't a close call distorted by variance: if two models were equivalent you'd expect ~25 wins.

Three specific readings:

1. **More capacity made things worse.** SVM and MLP are the two lowest-scoring models. With 19 positives, both have far more parameters than the problem can identify — the MLP's ROC-AUC of 0.629 is barely above chance, and its Brier score (0.085) is worse than the floor's 0.020. This is the tabular literature's point stated as a measurement, not an opinion: complexity has to be earned.
2. **k-NN's recall advantage is not an advantage.** 0.760 recall beats the rule's 0.702 — but it achieves that by flagging **15.0% of rows when the budget was 8.5%**, because distance-weighted k-NN scores are heavily tied and the threshold cannot land inside the budget. Precision (0.102 vs 0.172) and PR-AUC (0.139 vs 0.390) both show it. An operating point that spends nearly double the flag rate is not a comparison.
3. **The SVM's Brier score is the best of any real model (0.019)** while its ranking is third worst. Platt scaling on 15 positives produces confident, well-ordered-in-aggregate probabilities that are useless for ranking the rare class. Good calibration on a bad ranker is a trap worth recognising.

### None detection: the result that changed

`models.none_score` used 1 − max probability. Here four alternatives are compared on the same held-out-category task: max probability, decision margin, energy, Mahalanobis distance from the training feature distribution, and k-NN distance.

| Score | AUROC (6 runs) | Detection at 5% FPR |
|---|---|---|
| **Mahalanobis** | **0.864 – 0.890** | 20–27% |
| max probability | 0.572 – 0.645 | 3–13% |
| k-NN distance | 0.559 – 0.607 | 5–12% |
| decision margin | 0.336 – 0.428 | 0–2% |
| energy | 0.336 – 0.428 | 0–2% |

Three findings, in order of importance:

1. **Mahalanobis works, and nothing else does.** 0.86–0.89 AUROC against 0.34–0.65 for the classifier-head scores, and 20–27% detection inside a 5% false-positive budget. The earlier conclusion — that this problem had no usable none detector — was correct about *softmax scores* and wrong as a statement about the problem. Feature-space distance sees the unknown class; the penultimate-layer confidence does not.
2. **Energy and margin are numerically identical** (0.428 and 0.428, every run). That is not a coincidence: for a binary classifier the energy is `−log(1 + exp(f))`, a monotone function of the margin, so it cannot rank differently. Energy-based detection earns its reputation in multiclass settings; in binary it is the margin with extra steps. Likewise, the detector depends only on the preprocessing, so logistic and SVM produce byte-identical Mahalanobis scores — the representation carries the signal, not the classifier.
3. **The head scores point the wrong way.** AUROC below 0.5 means unknown rows look *more* like clicks than known rows do — consistent with the Issue 1 finding that held-out categories scored higher. A model trained only on closed-set classes has no reason to place a new class low; the confidence head is the wrong instrument.

**Caveat, stated plainly:** the strongest component of the Mahalanobis signal is almost certainly the one-hot category block. The encoder maps an unseen category to all zeros (`handle_unknown="ignore"`), so a held-out row is far from every training row *by construction*. That is legitimate behaviour for this data type — "nothing matches any known category" is a real signal — but it will not generalise to purely continuous features, where there is no flag to go missing. Reporting 0.89 AUROC without that caveat would overstate what was shown.

### Artifacts

Issue 2 adds `advanced_threshold_free.csv`, `advanced_operating.csv`, `advanced_folds.csv` (600 fold rows), `paired_vs_rule.csv` and `none_detectors.csv` to every run. The persistence checks in section 12 now cover 15 artifacts and fail the run if the advanced tables don't reload with their original row counts.

### Update to the Issue 1 benchmark

Issue 1 concluded "nothing in this sprint produces an open-set score above 0.65 AUROC". That was true of the score it tested. With a distance-based detector the open-set target is met — Mahalanobis at 0.86–0.89 — while the *click-prediction* target is still unmet, since no model beats the rule's 0.390 PR-AUC.

---

## Sprint 5, Issue 1: Cross-Validation and Hyperparameter Tuning

Code: `src/model_optimization.py`. Run `python main.py` section 13. Adds about 50s to the run.

### Why nested

Tuning on the folds you then report is how a project manufactures an improvement: the search keeps whichever configuration fits that split, and the reported score includes the picking. With 19 positives the bias is easily bigger than the real differences. So tuning happens on inner folds inside each outer training split, and the outer fold is scored once.

The prior-history rule is scored on those **same outer folds**, which makes the model-vs-reference margin paired rather than a comparison of numbers from two different CV runs. Two comparisons come out of one pass: tuned vs default, and tuned vs reference.

### Results (10 outer folds, 3 inner folds)

| Model | Tuned PR-AUC | Default | Gain | Folds better | Inner best | Outer − inner |
|---|---|---|---|---|---|---|
| logistic regression | **0.310 ± 0.125** | 0.243 | +0.067 | 6/10 | 0.310 | −0.000 |
| RBF SVM | 0.298 ± 0.175 | 0.117 | **+0.181** | **10/10** | 0.311 | −0.013 |
| random forest | 0.226 ± 0.109 | 0.182 | +0.044 | 9/10 | 0.262 | −0.037 |
| k-NN | 0.224 ± 0.191 | 0.047 | +0.177 | **10/10** | 0.246 | −0.022 |
| MLP (32-16) | 0.162 ± 0.153 | 0.040 | +0.122 | 7/10 | 0.151 | +0.011 |
| *reference: prior-history rule* | *0.386* | — | — | — | — | — |

### What the tuning actually bought

1. **The gain tracks how bad the default was, not how good the model is.** The two biggest improvements are on the two models with the worst defaults — SVM +0.181 and k-NN +0.177, each winning 10 of 10 folds. Logistic regression, whose default `C=1.0` is already reasonable, gains +0.067 and wins only 6 of 10. Tuning is mostly repairing bad starting points, which is worth doing once and worth doing sceptically.
2. **Nothing beats a one-line rule.** Best tuned model is 0.310 against the reference's 0.386 on identical folds. The gap (−0.076) is larger than the tie tolerance and larger than the tuned model's own advantage over its default.
3. **The search did not overfit the inner folds much.** Outer-minus-inner is −0.037 to +0.011, so nested CV is not showing a dramatic optimism gap here. The grids are deliberately small (4-6 candidates), which is the reason: a wide search on 19 positives would have produced a much prettier and much less true inner score.
4. **The parameter wins are mostly coin flips.** 12 of 19 parameter-value rows were chosen in under 80% of folds. Random forest's `max_depth` split 5/5 between `None` and `3` — that is not a tuned setting, it is noise. The consistent ones are informative though: `C=0.01` for logistic (8/10), `gamma=0.01` for the SVM (9/10), and `min_samples_leaf=5` for the forest (8/10) all point the same way — **the data wants heavy regularisation**, which is what you'd expect with 19 positives.

### Overfitting and underfitting

Learning curves, train vs cross-validated PR-AUC as rows are added:

| Train rows | LR train | LR CV | Gap | RF train | RF CV | Gap |
|---|---|---|---|---|---|---|
| 20 | 1.000 | 0.121 | 0.879 | 1.000 | 0.137 | 0.863 |
| 40 | 1.000 | 0.144 | 0.856 | 1.000 | 0.152 | 0.848 |
| 60 | 1.000 | 0.142 | 0.858 | 1.000 | 0.160 | 0.840 |
| 80 | 1.000 | 0.358 | 0.642 | 1.000 | 0.221 | 0.779 |
| 100 | 0.961 | 0.286 | 0.675 | 1.000 | 0.274 | 0.726 |

Both models memorise the training rows and generalise poorly — a gap of 0.64 to 0.88 that narrows only slowly as data is added. That is overfitting as a *structural property of the dataset*, not a hyperparameter mistake: no setting in either grid closes it, and the CV curve is still climbing at 100 rows, meaning the fix is more data rather than more tuning.

The MLP is the underfitting case by contrast — tuned 0.162 with both train and validation low relative to its capacity, and only 7/10 folds improved by tuning. A two-layer network is the wrong shape for 19 positives.

**Caveat on the learning curve.** With 1–2 positives per validation fold, average precision can only take values like 0, 0.5 and 1. The `std = 0.000` at the small sizes is that granularity, not stability, and the train score of exactly 1.000 means "the single positive was ranked first" rather than "perfect generalisation". The direction of the conclusion is sound; the precision of the numbers is not.

### Final selection

```
selected: prior-history rule
reason:   best tuned model (logistic regression) scores 0.310 PR-AUC against the
          reference's 0.386 (margin -0.076), which is inside the 0.02 tie tolerance or
          below its own std (0.125); the reference has no fitted parameters to go stale,
          so it wins the tie
```

The selection rule is deliberately conservative: a tuned model is only preferred if it beats the reference by more than a 0.02 tolerance *and* by more than its own fold-to-fold std. That stops a 0.01 win on one split from becoming "the final model", which at this sample size would be a story rather than a finding.

### Artifacts added

`nested_cv_folds.csv` (50 rows), `nested_cv_summary.csv`, `tuning_stability.csv`, `learning_curve_lr.csv`, `learning_curve_rf.csv`, and `final_selection.json`. The persistence checks now cover 21 artifacts; `final_selection.json` is also referenced in `latest.json` as `final_model`.
