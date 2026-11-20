"""
baseline.py - Linear regression baseline for Nothing But Net.

Predicts a player's points as a normal distribution:
    mean  (mu)    = linear regression on rolling features
    spread (sigma) = ONE fixed value for everyone (std of training errors)

Reads:  data/clean/player_games.parquet
Writes: models/baseline_scaler.pkl, models/baseline_model.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clean" / "players_features.parquet"
MODEL_DIR = ROOT / "models"

FEATURES = [
    "PTS_ROLL5", "PTS_ROLL10", "PTS_STD10",
    "MIN_ROLL5", "MIN_ROLL10",
    "FGA_ROLL5", "FGA_ROLL10", "FTA_ROLL5", "FTA_ROLL10",
    "REST_DAYS", "B2B", "HOME", "GAMES_PLAYED",
    "OPP_PTS_ALLOWED_ROLL10", "OPP_PACE_ROLL10", "OPP_DEF_RTG_ROLL10",
]
TARGET = "PTS"
LINE = "PROXY_LINE"


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
    df = pd.read_parquet(DATA)

    # --- split by season (time)
    train = df[df["SEASON"] <= "2023-24"]
    val = df[df["SEASON"] == "2024-25"]
    # test (2025-26 onward) is NOT touched here; only evaluate on it once, at the end

    # --- drop rows missing any model input, target or line
    cols = FEATURES + [TARGET, LINE]
    train = train.dropna(subset=cols).copy()
    val = val.dropna(subset=cols).copy()
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
    print("Validation (2024-25):")
    # naive reference: just predict the 10-game average, same fixed-sigma idea
    naive_mu_train = train["PTS_ROLL10"].to_numpy()
    naive_sigma = np.std(y_train - naive_mu_train)
    evaluate("naive (10-game avg)", y_val, val["PTS_ROLL10"].to_numpy(), naive_sigma, line_val)
    evaluate("linear regression", y_val, model.predict(X_val), sigma, line_val)

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
