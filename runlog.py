"""
runlog.py - Run folders, so every reported number can be traced back to a run.

Each call to start_run() creates a folder like:

    runs/2026-11-20_18-30-05_lstm/
        params.json    settings + seed + git commit + library versions
        log.txt        everything the script printed (same as the terminal)
        metrics.json   final numbers (written by save_metrics)
        ...            anything else the script saves there (plots, history.csv, weights)

Usage inside a training script:

    from runlog import start_run, save_metrics
    run_dir = start_run("baseline", {"features": FEATURES, "seed": 42})
    ...
    save_metrics(run_dir, {"val": {"mae": 4.66, ...}})
"""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"


class _Tee:
    """Write everything to the terminal AND to log.txt at the same time."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()          # so the log is complete even if the run crashes

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def _git_commit():
    """Commit hash of the code that produced this run ('-dirty' if there were uncommitted edits)."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=ROOT, text=True).strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _versions():
    import pandas, sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__,
         "pandas": pandas.__version__, "sklearn": sklearn.__version__}
    try:
        import torch
        v["torch"] = torch.__version__
    except ImportError:
        pass
    return v


def _to_jsonable(obj):
    """numpy numbers/arrays -> plain Python so json.dump doesn't fail."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def start_run(name, params):
    """Create runs/<timestamp>_<name>/, save params.json and start copying output to log.txt."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    info = {
        "run": run_dir.name,
        "script": Path(sys.argv[0]).name,
        "command": " ".join(sys.argv),
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "params": params,
    }
    with open(run_dir / "params.json", "w") as f:
        json.dump(_to_jsonable(info), f, indent=2)

    log_file = open(run_dir / "log.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"run folder: {run_dir.relative_to(ROOT)}")
    return run_dir


def save_metrics(run_dir, metrics):
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(_to_jsonable(metrics), f, indent=2)
    print(f"metrics saved to {(run_dir / 'metrics.json').relative_to(ROOT)}")
