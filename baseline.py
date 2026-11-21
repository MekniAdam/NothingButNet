"""
baseline.py - Linear regression baseline for Nothing But Net.

Predicts a player's points as a normal distribution:
    mean  (mu)    = linear regression on rolling features
    spread (sigma) = ONE fixed value for everyone (std of training errors)

Reads:  data/clean/players_features.parquet   (split + features defined in split.py)
Writes: runs/<time>_baseline/, models/baseline_scaler.pkl, models/baseline_model.pkl
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from runlog import save_metrics, start_run
# The split, feature list, target and line all come from split.py (one shared definition)
from split import (FEATURES, LINE, SEED, TARGET, TRAIN_LAST, VAL_SEASON,
                   load_players, split)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Linear regression baseline.")
    parser.add_argument("--frac", type=float, default=1.0,
                        help="fraction of players to use, e.g. 0.1 for a quick run (default: all)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="random seed (only used to pick players when --frac < 1)")
    return parser.parse_args()


# metrics
def evaluate(name, y, mu, sigma, line):
    """Print MAE, NLL, over/under hit rate and calibration (ECE)."""
    mae = np.mean(np.abs(y - mu))
    nll = -norm.logpdf(y, mu, sigma).mean()

    # Over/under vs the line. Skip pushes (points exactly equal to the line).
    p_over = 1 - norm.cdf(line, mu, sigma)
    went_over = (y > line).astype(int)
    no_push = y != line
    pick_over = p_over > 0.5
    hit_rate = np.mean(pick_over[no_push] == went_over[no_push].astype(bool))

    # ECE: when we say "70% chance of over", does it go over ~70% of the time?
    bins = np.linspace(0, 1, 11)
    which_bin = np.clip(np.digitize(p_over[no_push], bins) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        in_bin = which_bin == b
        if in_bin.any():
            gap = abs(p_over[no_push][in_bin].mean() - went_over[no_push][in_bin].mean())
            ece += in_bin.mean() * gap

    print(f"{name:<22} MAE={mae:5.2f}  NLL={nll:5.3f}  hit={hit_rate:6.1%}  ECE={ece:.3f}")
    return {"mae": mae, "nll": nll, "hit_rate": hit_rate, "ece": ece}


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    # Run folder: params.json + log.txt + metrics.json (see runlog.py)
    run_dir = start_run("baseline", {
        "model": "LinearRegression + one fixed sigma",
        "features": FEATURES, "target": TARGET, "line": LINE,
        "split": {"train": f"<= {TRAIN_LAST}", "val": VAL_SEASON},
        "frac": args.frac,
        # least squares itself is not random; the seed only picks players when frac < 1
        "seed": args.seed,
    })

    df = load_players(args.frac, args.seed)

    # --- split by season, keeping only usable rows (see split.py).
    # test is NOT used here; it is evaluated once, at the very end of the project.
    train, val, _ = split(df)
    print(f"train rows: {len(train):,}   val rows: {len(val):,}\n")

    # --- normalize: fit on TRAIN only
    scaler = StandardScaler().fit(train[FEATURES])
    X_train = scaler.transform(train[FEATURES])
    X_val = scaler.transform(val[FEATURES])
    y_train = train[TARGET].to_numpy(dtype=float)
    y_val = val[TARGET].to_numpy(dtype=float)
    line_val = val[LINE].to_numpy(dtype=float)

    # --- fit linear regression for the mean
    model = LinearRegression().fit(X_train, y_train)

    # --- fixed sigma = spread of the training errors
    sigma = np.std(y_train - model.predict(X_train))
    print(f"fixed sigma: {sigma:.2f} points\n")

    # --- evaluate on validation
    print(f"Validation ({VAL_SEASON}):")
    # naive reference: just predict the 10-game average, same fixed-sigma idea
    naive_mu_train = train["PTS_ROLL10"].to_numpy()
    naive_sigma = np.std(y_train - naive_mu_train)
    naive_res = evaluate("naive (10-game avg)", y_val, val["PTS_ROLL10"].to_numpy(),
                         naive_sigma, line_val)
    linreg_res = evaluate("linear regression", y_val, model.predict(X_val), sigma, line_val)
    save_metrics(run_dir, {
        "train_rows": len(train), "val_rows": len(val),
        "sigma": sigma, "naive_sigma": naive_sigma,
        "val": {"naive (10-game avg)": naive_res, "linear regression": linreg_res},
    })

    # --- which features matter? (coefficients are comparable because inputs are scaled)
    coefs = pd.Series(model.coef_, index=FEATURES).sort_values(key=abs, ascending=False)
    print("\nCoefficients (points per 1 std of feature):")
    print(coefs.round(2).to_string())

    # --- save for later use (same scaler MUST be reused for predictions)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "baseline_scaler.pkl")
    joblib.dump({"model": model, "sigma": sigma, "features": FEATURES},
                MODEL_DIR / "baseline_model.pkl")
    print(f"\nsaved to {MODEL_DIR}")
    # copy into the run folder too, so this exact model stays with its metrics
    joblib.dump(scaler, run_dir / "baseline_scaler.pkl")
    joblib.dump({"model": model, "sigma": sigma, "features": FEATURES},
                run_dir / "baseline_model.pkl")

    scaler = joblib.load("models/baseline_scaler.pkl")
    saved  = joblib.load("models/baseline_model.pkl")
    model, sigma, features = saved["model"], saved["sigma"], saved["features"]

    # What the scaler learned from TRAIN (mean and std of each feature)
    print(pd.DataFrame({"mean": scaler.mean_, "std": scaler.scale_}, index=features))

    # What the model learned
    print("intercept:", model.intercept_)
    print("sigma:", sigma)
    print(pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False))


if __name__ == "__main__":
    main()
