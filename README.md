# aiap-capstone-project

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
