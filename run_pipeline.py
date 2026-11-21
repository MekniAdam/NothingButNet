"""
run_pipeline.py - Run the whole Nothing But Net pipeline with one command.

Steps, in order (each one is a normal script you can also run by itself):
  1. setup_data.py   download raw NBA data        (only if data/raw is missing, or --download)
  2. build_teams.py  build features               -> data/clean/
  3. eda.py          data exploration plots       -> figures/
  4. baseline.py     linear regression baseline   -> runs/<time>_baseline/
  5. lstm.py         LSTM (mean + sigma)          -> runs/<time>_lstm/

Run from the project root:
    python run_pipeline.py                  # everything, all players
    python run_pipeline.py --frac 0.1       # quick check on 10% of players
    python run_pipeline.py --download       # also refresh the raw data first
    python run_pipeline.py --seed 7         # different seed for the models

The pipeline stops at the first step that fails.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "player_gamelogs"


def run(step, script, *args):
    """Run one script with the same Python as this one; stop everything if it fails."""
    cmd = [sys.executable, script, *[str(a) for a in args]]
    print(f"\n{'=' * 70}\n[{step}] {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT)        # cwd=ROOT: scripts use relative data paths
    if result.returncode != 0:
        sys.exit(f"\n[{step}] FAILED (exit code {result.returncode}); pipeline stopped.")
    elapsed = time.time() - start
    print(f"[{step}] done in {elapsed:.0f}s", flush=True)
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline.")
    parser.add_argument("--frac", type=float, default=1.0,
                        help="fraction of players for the models, e.g. 0.1 (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="seed for the models")
    parser.add_argument("--epochs", type=int, default=None, help="max LSTM epochs")
    parser.add_argument("--download", action="store_true",
                        help="re-run setup_data.py even if raw data already exists")
    parser.add_argument("--skip-eda", action="store_true", help="skip the EDA plots")
    args = parser.parse_args()

    times = {}
    if args.download or not RAW_DIR.exists():
        times["download"] = run("1/5 download", "setup_data.py")
    else:
        print("[1/5 download] skipped: data/raw already exists (use --download to refresh)")

    times["features"] = run("2/5 features", "build_teams.py")

    if args.skip_eda:
        print("[3/5 eda] skipped (--skip-eda)")
    else:
        times["eda"] = run("3/5 eda", "eda.py")

    model_args = ["--frac", args.frac, "--seed", args.seed]
    times["baseline"] = run("4/5 baseline", "baseline.py", *model_args)

    lstm_args = model_args + (["--epochs", args.epochs] if args.epochs else [])
    times["lstm"] = run("5/5 lstm", "lstm.py", *lstm_args)

    print(f"\n{'=' * 70}\nPipeline finished. Step times:")
    for step, sec in times.items():
        print(f"  {step:<10} {sec:6.0f}s")
    print("Results: newest folders in runs/ (metrics.json, learning_curve.png), plots in figures/")


if __name__ == "__main__":
    main()
