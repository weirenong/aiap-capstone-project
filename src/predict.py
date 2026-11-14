"""Batch scoring CLI.

Loads a persisted run (model + manifest + scoring configuration), validates an input CSV
against the training schema, and writes per-row click probabilities and a none/click flag.

    python -m src.predict --input new_impressions.csv
    python -m src.predict --input new_impressions.csv --run results/run_20261114_041514
    python -m src.predict --input new_impressions.csv --threshold 0.30 --output scored.csv

Exit codes: 0 success, 2 bad input (schema or unreadable file), 3 missing run artifacts.
The threshold defaults to the one recorded in the run at `scoring.json`; passing
`--threshold` overrides it, which is the right move when the operating budget changes but
the model does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.artifacts import load_json, load_latest_run, load_model
from src.preprocessing import FEATURES, TARGET


def _resolve_run(run_arg: str | None) -> Path:
    if run_arg:
        run_dir = Path(run_arg)
        if not run_dir.exists():
            print(f"run directory not found: {run_dir}", file=sys.stderr)
            raise SystemExit(3)
        return run_dir
    latest = load_latest_run()
    if latest is None:
        print("no runs found; run `python main.py` first or pass --run", file=sys.stderr)
        raise SystemExit(3)
    return latest


def _validate_input(frame: pd.DataFrame) -> list[str]:
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        print(
            f"input is missing required columns: {missing}\n"
            f"expected: {list(FEATURES)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if frame.empty:
        print("input has no rows", file=sys.stderr)
        raise SystemExit(2)
    return missing


def score(frame: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    """Attach probability and flag columns. Keeps the input columns for traceability."""
    out = frame.copy()
    probabilities = model.predict_proba(frame[list(FEATURES)])[:, 1]
    out["p_click"] = probabilities.round(6)
    out["flagged"] = (probabilities >= threshold).astype(int)
    return out


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load the run, score the input and return an exit code."""
    parser = argparse.ArgumentParser(description="Score ad impressions with a persisted run.")
    parser.add_argument("--input", required=True, help="CSV of rows to score")
    parser.add_argument("--run", default=None, help="run directory (default: latest)")
    parser.add_argument("--output", default=None, help="output CSV (default: stdout summary only)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the run's recorded threshold",
    )
    parser.add_argument(
        "--drop-target",
        action="store_true",
        help=f"drop the {TARGET!r} column if present (it is a label, not a feature)",
    )
    args = parser.parse_args(argv)

    run_dir = _resolve_run(args.run)
    try:
        manifest = load_json(run_dir, "manifest")
        scoring = load_json(run_dir, "scoring")
    except FileNotFoundError as exc:
        print(f"missing run artifact: {exc}", file=sys.stderr)
        return 3

    threshold = args.threshold if args.threshold is not None else float(scoring["threshold"])
    model = load_model(run_dir, scoring["model_artifact"])

    frame = pd.read_csv(args.input)
    if args.drop_target and TARGET in frame.columns:
        frame = frame.drop(columns=[TARGET])
    _validate_input(frame)

    scored = score(frame, model, threshold)

    print(f"run:        {run_dir}")
    print(f"model:      {scoring['model_artifact']} ({scoring.get('model_type', 'unknown')})")
    print(f"data sha256:{manifest.get('data', {}).get('sha256', 'n/a')[:16]}...")
    print(f"rows:       {len(scored)}")
    print(f"threshold:  {threshold:.4f} (target flag rate {scoring['flag_rate']:.1%})")
    print(f"flagged:    {int(scored['flagged'].sum())} ({scored['flagged'].mean():.1%})")
    print(f"score range:{scored['p_click'].min():.4f} - {scored['p_click'].max():.4f}")

    if args.output:
        scored.to_csv(args.output, index=False)
        print(f"written:    {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
