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
