"""
lstm.py - LSTM with a probabilistic head for Nothing But Net.

Goal: fix the baseline's main weakness, the FIXED sigma.
This model predicts a separate mean (mu) AND spread (sigma) for every player-game,
trained with Gaussian negative log-likelihood (NLL).

Inputs for each player-game:
  * sequence: the player's last N games (points, minutes, shots, home, rest)
  * context:  what we know about TONIGHT (home, rest, opponent defense, history length)

Reads:  data/clean/players_features.parquet
        models/baseline_*.pkl   (optional, for a side-by-side comparison)
Writes: models/lstm.pt

Run from the project root (same folder as baseline.py):
    python lstm.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reuse the exact same metrics, features and row filter as the baseline,
# so the comparison is apples to apples.
from baseline import FEATURES, LINE, TARGET, evaluate

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clean" / "players_features.parquet"
MODEL_DIR = ROOT / "models"

# ------------------------------------------------------------------ settings
SEQ_LEN = 10            # how many previous games the LSTM looks at
SEQ_COLS = ["PTS", "MIN", "FGA", "FTA", "HOME", "REST_DAYS"]   # per past game
CTX_COLS = [            # known before tonight's tip-off
    "HOME", "REST_DAYS", "B2B", "GAMES_PLAYED_LOG",
    "OPP_PTS_ALLOWED_ROLL10", "OPP_PACE_ROLL10", "OPP_DEF_RTG_ROLL10",
]
HIDDEN = 128
BATCH = 512
LR = 1e-3
MAX_EPOCHS = 30
PATIENCE = 3            # stop if val NLL doesn't improve for this many epochs
SEED = 42


# ------------------------------------------------------------------ data
def build_sequences(df):
    """For every row, stack the player's previous SEQ_LEN games -> (rows, SEQ_LEN, cols).

    shift(k) = the game k games ago for the same player, so tonight's game is never
    included (no leakage). Players with fewer than SEQ_LEN past games get NaN at the
    front, which becomes zero padding plus a mask channel saying 'no game here'.
    """
    g = df.groupby("PLAYER_ID")[SEQ_COLS]
    lags = [g.shift(k).to_numpy(dtype=np.float32) for k in range(SEQ_LEN, 0, -1)]
    return np.stack(lags, axis=1)          # oldest game first, most recent last


def scale(arr, mean, std):
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0).astype(np.float32)   # missing -> average (0)


def prepare(df):
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    df["GAMES_PLAYED_LOG"] = np.log1p(df["GAMES_PLAYED"])

    seq_raw = build_sequences(df)          # built on ALL rows so history is complete
    has_game = ~np.isnan(seq_raw[:, :, :1])                   # (rows, SEQ_LEN, 1)

    # Same rows as the baseline (needs features, target and a line).
    usable = df[FEATURES + [TARGET, LINE]].notna().all(axis=1)
    train_m = usable & (df["SEASON"] <= "2023-24")
    val_m = usable & (df["SEASON"] == "2024-25")
    # test (2025-26 onward) is NOT touched here; evaluate on it once, at the very end

    # --- scaling statistics from TRAIN rows only
    seq_mean = np.nanmean(df.loc[train_m, SEQ_COLS].to_numpy(float), axis=0)
    seq_std = np.nanstd(df.loc[train_m, SEQ_COLS].to_numpy(float), axis=0) + 1e-6
    ctx_mean = df.loc[train_m, CTX_COLS].mean().to_numpy()
    ctx_std = df.loc[train_m, CTX_COLS].std().to_numpy() + 1e-6
    y_mean = df.loc[train_m, TARGET].mean()
    y_std = df.loc[train_m, TARGET].std()

    seq = np.concatenate([scale(seq_raw, seq_mean, seq_std),
                          has_game.astype(np.float32)], axis=2)
    ctx = scale(df[CTX_COLS].to_numpy(float), ctx_mean, ctx_std)
    y = df[TARGET].to_numpy(dtype=np.float32)

    stats = dict(seq_mean=seq_mean, seq_std=seq_std, ctx_mean=ctx_mean,
                 ctx_std=ctx_std, y_mean=y_mean, y_std=y_std)
    return df, seq, ctx, y, train_m.to_numpy(), val_m.to_numpy(), stats


# ------------------------------------------------------------------ model
class PropLSTM(nn.Module):
    def __init__(self, n_seq, n_ctx, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_ctx, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 2),                     # -> [mu, raw_sigma]
        )

    def forward(self, seq, ctx):
        _, (h, _) = self.lstm(seq)                # h[-1] = summary of the last N games
        out = self.head(torch.cat([h[-1], ctx], dim=1))
        mu = out[:, 0]
        sigma = nn.functional.softplus(out[:, 1]) + 1e-3   # softplus keeps sigma > 0
        return mu, sigma                          # both in STANDARDIZED target units


def gaussian_nll(y, mu, sigma):
    return (torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2).mean()


@torch.no_grad()
def predict(model, seq, ctx, stats, device):
    model.eval()
    mus, sigmas = [], []
    for i in range(0, len(seq), 4096):
        s = torch.from_numpy(seq[i:i + 4096]).to(device)
        c = torch.from_numpy(ctx[i:i + 4096]).to(device)
        mu, sigma = model(s, c)
        mus.append(mu.cpu().numpy())
        sigmas.append(sigma.cpu().numpy())
    # convert BOTH back to points (forgetting sigma here is a classic bug)
    mu = np.concatenate(mus) * stats["y_std"] + stats["y_mean"]
    sigma = np.concatenate(sigmas) * stats["y_std"]
    return mu, sigma


# ------------------------------------------------------------------ main
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    df, seq, ctx, y, train_m, val_m, stats = prepare(pd.read_parquet(DATA))
    print(f"train rows: {train_m.sum():,}   val rows: {val_m.sum():,}")

    y_scaled = ((y - stats["y_mean"]) / stats["y_std"]).astype(np.float32)
    train_ds = TensorDataset(torch.from_numpy(seq[train_m]), torch.from_numpy(ctx[train_m]),
                             torch.from_numpy(y_scaled[train_m]))
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    model = PropLSTM(seq.shape[2], ctx.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}\n")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    y_val, line_val = y[val_m], df.loc[val_m, LINE].to_numpy(float)
    best_nll, best_state, bad_epochs = np.inf, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        for s, c, t in loader:
            s, c, t = s.to(device), c.to(device), t.to(device)
            mu, sigma = model(s, c)
            loss = gaussian_nll(t, mu, sigma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(t)

        mu_v, sig_v = predict(model, seq[val_m], ctx[val_m], stats, device)
        val_nll = np.mean(np.log(sig_v) + 0.5 * ((y_val - mu_v) / sig_v) ** 2
                          + 0.5 * np.log(2 * np.pi))
        print(f"epoch {epoch:2d}  train loss {total / len(train_ds):.4f}  val NLL {val_nll:.4f}")

        if val_nll < best_nll - 1e-4:
            best_nll, bad_epochs = val_nll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("early stopping (val NLL stopped improving)")
                break

    model.load_state_dict(best_state)
    mu_v, sig_v = predict(model, seq[val_m], ctx[val_m], stats, device)

    # ------------------------------------------------------------ results
    val = df.loc[val_m]
    print("\nValidation (2024-25):")
    results = {"LSTM": (mu_v, sig_v)}
    base_path = MODEL_DIR / "baseline_model.pkl"
    if base_path.exists():
        saved = joblib.load(base_path)
        scaler = joblib.load(MODEL_DIR / "baseline_scaler.pkl")
        mu_b = saved["model"].predict(scaler.transform(val[saved["features"]]))
        results = {"linear regression": (mu_b, np.full_like(mu_b, saved["sigma"])), **results}

    for name, (mu, sig) in results.items():
        evaluate(name, y_val, mu, sig, line_val)

    # Does a per-player sigma fix calibration where the fixed sigma failed?
    print("\nBy player type:")
    tiers = {"low scorers (<10)": val["PTS_ROLL10"] < 10,
             "mid (10-20)": val["PTS_ROLL10"].between(10, 20),
             "high scorers (>20)": val["PTS_ROLL10"] > 20}
    for tier, m in tiers.items():
        m = m.to_numpy()
        print(f"  {tier}  ({m.sum():,} rows)")
        for name, (mu, sig) in results.items():
            evaluate(f"    {name}", y_val[m], mu[m], sig[m], line_val[m])

    print("\nLSTM sigma (points):")
    for tier, m in tiers.items():
        print(f"  {tier:<20} median sigma = {np.median(sig_v[m.to_numpy()]):.2f}")

    # ------------------------------------------------------------ save
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "stats": stats,
                "config": dict(SEQ_LEN=SEQ_LEN, SEQ_COLS=SEQ_COLS, CTX_COLS=CTX_COLS,
                               HIDDEN=HIDDEN)},
               MODEL_DIR / "lstm.pt")
    print(f"\nsaved to {MODEL_DIR / 'lstm.pt'}")


if __name__ == "__main__":
    main()


# =============================================================================
# HOW TO READ THE OUTPUT
# =============================================================================
#
# device: cuda / cpu
#     Whether training runs on your GPU. "cpu" on a machine with an RTX card means
#     PyTorch was installed without CUDA support; reinstall it from pytorch.org.
#
# train rows / val rows
#     Should match baseline.py exactly. Same rows means a fair comparison.
#
# parameters
#     Size of the model. Your proposal says ~50k-300k (excluding DistilBERT).
#
# epoch N  train loss ...  val NLL ...
#     train loss = Gaussian NLL on training data, in standardized units (it can be
#                  negative; that's normal for NLL).
#     val NLL    = NLL on 2024-25, in POINTS, the same scale as baseline.py.
#     Healthy run: both drop for a few epochs, then val NLL flattens.
#     If train loss keeps dropping while val NLL rises -> overfitting; early stopping
#     then keeps the best epoch, not the last one.
#
# Validation (2024-25): one line per model, same metrics as baseline.py
#     MAE  - average miss in points (judges mu only). Lower is better.
#     NLL  - quality of the whole distribution (mu + sigma). Lower is better.
#            This is the main metric: the LSTM should beat linear regression here.
#     hit  - % correct over/under picks vs. the PROXY line (an easy line; not a
#            betting result).
#     ECE  - how far stated probabilities are from what actually happened,
#            e.g. 0.03 = off by ~3 percentage points. Lower is better.
#
# By player type
#     The same metrics split by recent scoring. The baseline uses ONE sigma for
#     everyone, so expect its ECE/NLL to be worst for low or high scorers.
#     If the LSTM's per-player sigma works, its ECE should be lower in those groups.
#     This is the clearest evidence for the project's main claim.
#
# LSTM sigma (points)
#     The typical spread the LSTM predicts for each group. Unlike the baseline's
#     single value (~5.9), these should DIFFER: small for low scorers, larger for
#     high scorers. If all three are about the same, sigma isn't adapting.
#
# saved to models/lstm.pt
#     Weights + scaling statistics + settings. To predict later, rebuild inputs
#     with the SAME columns and scale them with these saved stats (never refit).
# =============================================================================
