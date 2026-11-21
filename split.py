"""
split.py - ONE place that defines the data split and the model inputs.

baseline.py, lstm.py and eda.py all import from here, so they always use the
same seasons, the same features and the same row filter. To change the split
or the feature list, change it HERE only.

Split (by season, no look-ahead):
    train       seasons up to and including TRAIN_LAST
    validation  VAL_SEASON         (early stopping, model selection, comparisons)
    test        TEST_FIRST onward  (evaluate ONCE, at the very end of the project)

A row is "usable" when it has every model input, the target and the line.

Run it on its own to print the split sizes:
    python split.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clean" / "players_features.parquet"   # made by build_teams.py

# ------------------------------------------------------------------ seasons
TRAIN_LAST = "2023-24"
VAL_SEASON = "2024-25"
TEST_FIRST = "2025-26"
# Season labels look like "2016-17", so plain string comparison sorts them by year.

# ------------------------------------------------------------------ model inputs
FEATURES = [
    "PTS_ROLL5", "PTS_ROLL10", "PTS_STD10",
    "MIN_ROLL5", "MIN_ROLL10",
    "FGA_ROLL5", "FGA_ROLL10", "FTA_ROLL5", "FTA_ROLL10",
    "REST_DAYS", "B2B", "HOME", "GAMES_PLAYED",
    "OPP_PTS_ALLOWED_ROLL10", "OPP_PACE_ROLL10", "OPP_DEF_RTG_ROLL10",
]
TARGET = "PTS"           # what we predict
LINE = "PROXY_LINE"      # the over/under line we evaluate against (proxy until real lines)
SEED = 42


# ------------------------------------------------------------------ functions
def subsample_players(df, frac, seed=SEED):
    """Keep a random `frac` of PLAYERS (all of their games), for quick test runs.

    Sampling whole players (not single rows) keeps every kept player's game
    history complete, so rolling features and LSTM sequences stay valid.
    The same frac + seed always picks the same players, in every script.
    """
    if frac >= 1.0:
        return df
    players = np.sort(df["PLAYER_ID"].unique())
    rng = np.random.default_rng(seed)
    keep = rng.choice(players, size=max(1, int(len(players) * frac)), replace=False)
    return df[df["PLAYER_ID"].isin(keep)]


def load_players(frac=1.0, seed=SEED):
    """Read the feature table (optionally a fraction of players)."""
    df = subsample_players(pd.read_parquet(DATA), frac, seed)
    if frac < 1.0:
        print(f"using {frac:.0%} of players: {df['PLAYER_ID'].nunique():,} players")
    return df


def usable_mask(df):
    """True where the row has every model input, the target and the line."""
    return df[FEATURES + [TARGET, LINE]].notna().all(axis=1)


def split_masks(df, usable_only=True):
    """Boolean masks (aligned with df) for the train / val / test rows.

    Masks rather than copies, because lstm.py needs ALL rows to build each
    player's game history, and only then picks which rows to train on.
    """
    keep = usable_mask(df) if usable_only else pd.Series(True, index=df.index)
    return {
        "train": keep & (df["SEASON"] <= TRAIN_LAST),
        "val": keep & (df["SEASON"] == VAL_SEASON),
        "test": keep & (df["SEASON"] >= TEST_FIRST),
    }


def split(df, usable_only=True):
    """(train, val, test) DataFrames. Don't touch test until the final evaluation."""
    m = split_masks(df, usable_only)
    return df[m["train"]].copy(), df[m["val"]].copy(), df[m["test"]].copy()


def split_name(season):
    """'train' / 'val' / 'test' for a season label, e.g. for plot colours."""
    if season <= TRAIN_LAST:
        return "train"
    return "val" if season == VAL_SEASON else "test"


if __name__ == "__main__":
    df = pd.read_parquet(DATA)
    all_m = split_masks(df, usable_only=False)
    use_m = split_masks(df)
    print(f"{'split':<6} {'seasons':<22} {'all rows':>10} {'usable':>10}")
    seasons = {"train": f"... -> {TRAIN_LAST}", "val": VAL_SEASON, "test": f"{TEST_FIRST} -> ..."}
    for name in ["train", "val", "test"]:
        print(f"{name:<6} {seasons[name]:<22} {all_m[name].sum():>10,} {use_m[name].sum():>10,}")
