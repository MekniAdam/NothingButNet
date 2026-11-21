"""
eda.py - Data exploration plots for Nothing But Net.

Reads:  data/clean/players_features.parquet   (run build_teams.py first)
Writes: figures/eda_*.png  and  figures/eda_summary.txt

Run from the project root:
    python eda.py

Only the TRAINING seasons (see split.py) are used for anything that could
influence modelling choices, so validation/test seasons stay unseen.
The only exception is plot 1 (rows per season), which just counts rows.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # save to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Same split, features and row filter as the models (see split.py)
from split import DATA, FEATURES, LINE, TARGET, split_masks, split_name, usable_mask

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"


def save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)
    train = df[split_masks(df, usable_only=False)["train"]]    # all training-season rows
    usable = df[split_masks(df)["train"]]                      # rows the models train on
    summary = []                                               # lines for eda_summary.txt

    # ------------------------------------------------------------ 1. rows per season
    # How much data each split has, and how many rows survive the feature filter.
    all_rows = df.groupby("SEASON").size()
    kept = df[usable_mask(df)].groupby("SEASON").size()
    colors = {"train": "C0", "val": "C1", "test": "C3"}
    split_color = {s: colors[split_name(s)] for s in all_rows.index}
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(all_rows))
    ax.bar(x, all_rows.values, color="lightgray", label="all player-games")
    ax.bar(x, kept.reindex(all_rows.index).fillna(0).values,
           color=[split_color[s] for s in all_rows.index], label="usable (no missing features)")
    ax.set_xticks(x, all_rows.index, rotation=45)
    ax.set_ylabel("player-games")
    ax.set_title("Rows per season  (blue = train, orange = val, red = test)")
    ax.set_ylim(0, all_rows.max() * 1.25)                  # leave room for the legend
    ax.legend(loc="upper left", ncol=2)
    save(fig, "eda_1_rows_per_season.png")

    # ------------------------------------------------------------ 2. points distribution
    # Is a Gaussian a reasonable shape for points? (Skew, pile-up at 0.)
    pts = usable[TARGET]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pts, bins=np.arange(0, 61), density=True, color="C0", alpha=0.7,
            label="actual points")
    grid = np.linspace(-10, 60, 300)
    ax.plot(grid, norm.pdf(grid, pts.mean(), pts.std()), color="C3",
            label=f"Gaussian with same mean/std ({pts.mean():.1f}, {pts.std():.1f})")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("points in a game")
    ax.set_ylabel("share of player-games")
    ax.set_title("Distribution of points (training seasons)")
    ax.legend()
    save(fig, "eda_2_points_distribution.png")
    summary += [
        f"points: mean {pts.mean():.2f}, median {pts.median():.1f}, std {pts.std():.2f}, "
        f"skew {pts.skew():.2f}",
        f"share of games with 0 points: {(pts == 0).mean():.1%}",
        f"Gaussian fit puts {norm.cdf(0, pts.mean(), pts.std()):.1%} of its mass below 0 points",
    ]

    # ------------------------------------------------------------ 3. spread grows with scoring
    # The key plot for the project: if the miss (PTS - recent average) gets wider for
    # higher scorers, one fixed sigma for everyone (the baseline) can't be calibrated.
    resid = usable[TARGET] - usable["PTS_ROLL10"]
    bins = pd.cut(usable["PTS_ROLL10"], bins=np.arange(0, 41, 2.5))
    by_bin = resid.groupby(bins, observed=True).agg(["std", "size"])
    by_bin = by_bin[by_bin["size"] >= 200]                    # skip tiny, noisy bins
    centers = [b.mid for b in by_bin.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, by_bin["std"], marker="o", label="std of (points - 10-game avg)")
    ax.axhline(resid.std(), color="C3", linestyle="--",
               label=f"one fixed value for everyone ({resid.std():.2f})")
    ax.set_xlabel("player's 10-game scoring average (points)")
    ax.set_ylabel("spread of the miss (points)")
    ax.set_title("Uncertainty grows with scoring level (training seasons)")
    ax.legend()
    save(fig, "eda_3_spread_by_scoring_level.png")
    summary.append("spread (std of points - 10-game avg) by scoring level:")
    summary += [f"  avg {b.left:>4.1f}-{b.right:<4.1f}: std {r['std']:.2f}  ({int(r['size']):,} rows)"
                for b, r in by_bin.iterrows()]

    # ------------------------------------------------------------ 4. over/under vs proxy line
    # How often players go over the proxy line, per scoring level. Near 50% means the
    # proxy line is "fair"; a trend means the line is biased for some players.
    no_push = usable[usable[TARGET] != usable[LINE]]
    over = (no_push[TARGET] > no_push[LINE])
    over_by_bin = over.groupby(pd.cut(no_push["PTS_ROLL10"], bins=np.arange(0, 41, 2.5)),
                               observed=True).agg(["mean", "size"])
    over_by_bin = over_by_bin[over_by_bin["size"] >= 200]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([b.mid for b in over_by_bin.index], over_by_bin["mean"], marker="o")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel("player's 10-game scoring average (points)")
    ax.set_ylabel("share of games OVER the proxy line")
    ax.set_title("Over rate vs proxy line (training seasons, pushes removed)")
    save(fig, "eda_4_over_rate_vs_proxy_line.png")
    summary.append(f"overall over-rate vs proxy line: {over.mean():.1%} "
                   f"({len(no_push):,} games, pushes removed)")

    # ------------------------------------------------------------ 5. context effects
    # Average miss vs recent form for home/away and rest days: do these features matter?
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    home = resid.groupby(usable["HOME"]).agg(["mean", "sem"])
    axes[0].bar(["away", "home"], home["mean"], yerr=1.96 * home["sem"], color=["C1", "C0"])
    axes[0].set_title("Home vs away")
    axes[0].set_ylabel("avg (points - 10-game avg)")
    rest = resid.groupby(usable["REST_DAYS"].clip(upper=4)).agg(["mean", "sem"])
    axes[1].bar([f"{int(r)}" + ("+" if r == 4 else "") for r in rest.index],
                rest["mean"], yerr=1.96 * rest["sem"], color="C2")
    axes[1].set_title("Days of rest (1 = back-to-back)")
    axes[1].set_xlabel("rest days")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
    fig.suptitle("Context effects on scoring vs recent form (95% CI, training seasons)")
    save(fig, "eda_5_context_effects.png")

    # ------------------------------------------------------------ 6. missing values
    # Which features are missing most, and so drop the most rows?
    miss = train[FEATURES + [LINE]].isna().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(miss.index, miss.values * 100, color="C0")
    ax.set_xlabel("% of training player-games missing")
    ax.set_title("Missing values per feature (training seasons)")
    save(fig, "eda_6_missing_values.png")
    summary.append(f"training rows: {len(train):,}  usable after dropping missing: "
                   f"{len(usable):,} ({len(usable) / len(train):.1%})")

    # ------------------------------------------------------------ summary
    (FIG_DIR / "eda_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n" + "\n".join(summary))
    print(f"\nsummary saved to {(FIG_DIR / 'eda_summary.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
