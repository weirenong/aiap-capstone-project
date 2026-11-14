"""Durable results and model artifacts.

Every run writes a self-contained directory under ``results/``:

    results/
      latest.json                  # pointer + headline metrics of the newest run
      run_20261114_101500/
        manifest.json              # versions, git commit, data hash, seeds
        index.md                   # human summary of what's in the run
        data_quality.json
        cv_folds.csv               # per-fold, per-operating-point rows
        cv_summary.csv             # mean/std across folds
        baseline_threshold_free.csv
        baseline_operating.csv
        feature_comparison.csv
        feature_stability.csv
        open_set_results.csv
        open_set_reports.json
        baseline_logistic.joblib   # fitted pipeline, loadable with load_model()

CSV for anything tabular, JSON for anything structured or nested, joblib for the fitted
pipeline. No parquet: it would add a pyarrow dependency for no benefit at this size.

The point is that a run is reproducible and comparable later without re-running
everything, which is the gap this module exists to close.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
LATEST_POINTER = "latest.json"


# --------------------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------------------


def new_run_dir(base: Path | str = RESULTS_ROOT, label: str = "run") -> Path:
    """Create a timestamped run directory and return it."""
    base = Path(base)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{label}_{stamp}"
    suffix = 1
    while run_dir.exists():  # two runs inside the same second
        run_dir = base / f"{label}_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def list_runs(base: Path | str = RESULTS_ROOT) -> list[Path]:
    """All run directories, oldest first."""
    base = Path(base)
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def load_latest_run(base: Path | str = RESULTS_ROOT) -> Path | None:
    """Resolve the newest run, using the pointer file if it is present."""
    base = Path(base)
    pointer = base / LATEST_POINTER
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text())
            candidate = base / payload["run_id"]
            if candidate.exists():
                return candidate
        except (json.JSONDecodeError, KeyError):
            pass
    runs = list_runs(base)
    return runs[-1] if runs else None


# --------------------------------------------------------------------------------------
# Serialisation helpers
# --------------------------------------------------------------------------------------


def _jsonable(value):
    """Convert numpy, pandas and dataclass values into plain JSON types."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def save_frame(frame: pd.DataFrame, name: str, run_dir: Path) -> dict[str, str]:
    """Write a table as CSV plus a JSON records file."""
    run_dir = Path(run_dir)
    csv_path = run_dir / f"{name}.csv"
    json_path = run_dir / f"{name}.json"

    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(_jsonable(frame.to_dict(orient="records")), indent=2))

    return {"csv": str(csv_path.relative_to(REPO_ROOT)), "json": str(json_path.relative_to(REPO_ROOT))}


def save_json(payload, name: str, run_dir: Path) -> str:
    """Write a dict / dataclass / list as indented JSON."""
    path = Path(run_dir) / f"{name}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2))
    return str(path.relative_to(REPO_ROOT))


def save_model(model, name: str, run_dir: Path) -> str:
    """Persist a fitted estimator or pipeline with joblib."""
    path = Path(run_dir) / f"{name}.joblib"
    joblib.dump(model, path)
    return str(path.relative_to(REPO_ROOT))


def load_frame(run_dir: Path, name: str) -> pd.DataFrame:
    """Read <name>.csv from a run directory as a DataFrame."""
    return pd.read_csv(Path(run_dir) / f"{name}.csv")


def load_json(run_dir: Path, name: str):
    """Read <name>.json from a run directory."""
    return json.loads((Path(run_dir) / f"{name}.json").read_text())


def load_model(run_dir: Path, name: str):
    """Load a fitted estimator or pipeline from a run directory."""
    return joblib.load(Path(run_dir) / f"{name}.joblib")


def compare_runs(run_a: Path | str, run_b: Path | str, metric: str = "pr_auc_mean") -> pd.DataFrame:
    """Diff one summary metric between two runs, matched on model and flag rate.

    This is the reason the tables are persisted at all: comparing two experiments should
    be a read, not a re-run.
    """
    a = pd.read_csv(Path(run_a) / "cv_summary.csv")
    b = pd.read_csv(Path(run_b) / "cv_summary.csv")
    keys = ["model", "flag_rate_target"]

    merged = a[keys + [metric]].merge(b[keys + [metric]], on=keys, suffixes=("_a", "_b"))
    merged["delta"] = merged[f"{metric}_b"] - merged[f"{metric}_a"]
    return merged.sort_values("delta", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def file_digest(path: Path | str) -> dict:
    """Size and sha256 of a file, so a run can be tied to exact input bytes."""
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.relative_to(REPO_ROOT)) if REPO_ROOT in path.parents else str(path),
            "bytes": path.stat().st_size,
            "sha256": digest}


def _versions() -> dict:
    import sklearn

    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    try:
        import imblearn

        versions["imbalanced-learn"] = imblearn.__version__
    except ImportError:  # pragma: no cover
        pass
    return versions


def write_manifest(run_dir: Path, *, config: dict | None = None, data_path=None) -> dict:
    """Record everything needed to say what produced this run."""
    manifest = {
        "run_id": Path(run_dir).name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "config": _jsonable(config or {}),
    }
    if data_path is not None:
        manifest["data"] = file_digest(data_path)

    path = Path(run_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


def write_pointer(run_dir: Path, headline: dict, base: Path | str = RESULTS_ROOT) -> str:
    """Write ``results/latest.json`` so the newest run is discoverable."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": Path(run_dir).name,
        "path": str(Path(run_dir).relative_to(REPO_ROOT)),
        "headline": _jsonable(headline),
    }
    path = base / LATEST_POINTER
    path.write_text(json.dumps(payload, indent=2))
    return str(path.relative_to(REPO_ROOT))


def write_index(run_dir: Path, summary: dict) -> str:
    """A short markdown index so a human can read the run without opening six CSVs."""
    lines = [f"# Run {Path(run_dir).name}", ""]
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.append(f"## {key.replace('_', ' ').title()}")
            lines.extend(f"- **{k}**: {v}" for k, v in value.items())
            lines.append("")
        else:
            lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("Artifacts: see `manifest.json`, `*.csv`, `*.json`, `*.joblib` in this folder.")
    path = Path(run_dir) / "index.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path.relative_to(REPO_ROOT))


__all__ = [
    "RESULTS_ROOT",
    "new_run_dir",
    "list_runs",
    "load_latest_run",
    "save_frame",
    "save_json",
    "save_model",
    "load_frame",
    "load_json",
    "load_model",
    "compare_runs",
    "file_digest",
    "write_manifest",
    "write_pointer",
    "write_index",
]
