# Documentation

Documentation for the ad click prediction capstone, framed as a **"None" classification
problem**: the model must be able to answer *"none of my known classes"* rather than always
picking one.

## Read this first

| If you are… | Start with | Then |
|---|---|---|
| Reviewing the project | [01_project_report.md](01_project_report.md) | [03_model_card.md](03_model_card.md) |
| Integrating or extending the code | [02_api_reference.md](02_api_reference.md) | [05_reproducibility.md](05_reproducibility.md) |
| Running it | [04_deployment_guide.md](04_deployment_guide.md) | [02_api_reference.md](02_api_reference.md) |
| Checking a result | [05_reproducibility.md](05_reproducibility.md) | `results/latest.json` |

## The documents

| File | Contents |
|---|---|
| [01_project_report.md](01_project_report.md) | Full report: problem framing, data audit, features, models, results, error analysis, limitations |
| [02_api_reference.md](02_api_reference.md) | Every module and public symbol, with signatures, arguments, return values and gotchas |
| [03_model_card.md](03_model_card.md) | Model card: intended use, metrics, calibration, ethical considerations, out-of-scope uses |
| [04_deployment_guide.md](04_deployment_guide.md) | Install, regenerate, batch scoring, input/output contracts, thresholds, monitoring, retraining |
| [05_reproducibility.md](05_reproducibility.md) | Environment, seeds, determinism evidence, artifact verification, failure modes |

## Status at a glance

- **Data**: 1,000 rows → 955 after removing under-18 records; 19 positives (2.0%).
- **Target**: `clicked_ad`, where 0 is the none class and 1 is a click.
- **Best model**: not a learned model. The prior-history rule (`previous_clicks ≤ 0.5`)
  reaches PR-AUC 0.390 and ROC-AUC 0.913; the best tuned classifier reaches 0.310.
- **Best none detector**: Mahalanobis distance, AUROC 0.864–0.890 on held-out ad
  categories — the one place where the None framing produces a usable result.
- **Calibration**: class-weighted models are badly calibrated (Brier 0.095 versus the
  always-none floor's 0.020) and should not be used as probability estimates without
  post-hoc calibration.
- **Verification**: `python main.py` runs 42 assertions and writes a 22-artifact run
  directory; the working tree reproduces to identical metrics.

## Documentation conventions

Numbers in these documents come from the artifacts under `results/`, not from memory. Where
a claim is a judgement rather than a measurement it is marked as such. Where a result is
inside its own standard deviation it is described as unresolved rather than as a finding —
with 19 positives that distinction is the difference between a result and a story.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                                  # full harness, ~2.5 minutes, writes results/
python -m src.predict --input rows.csv --output scored.csv
```
