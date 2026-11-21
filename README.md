# Nothing But Net

**Probabilistic NBA player-points prediction for over/under prop lines.**

ARTE 4110U – Deep Learning Architecture and Algorithms, Ontario Tech University, Fall 2026
Team: Adam Mekni, Elvis Sikapi, Madhav Patel

---

## What this project does

Sportsbooks post "prop lines" such as *"LeBron James over/under 24.5 points"*. Instead of predicting a single number, we predict each player's points as a **probability distribution**: a Gaussian with a mean **μ** (expected points) and a spread **σ** (uncertainty). From that distribution we get the **probability the player goes over the line**.

The core question is whether a neural network that learns a **per-player, per-game σ** gives better-calibrated probabilities than a baseline that uses one fixed σ for everyone. Later, injury and news signals will act as a "temperature" that widens σ when there is extra uncertainty.

| | Baseline (`baseline.py`) | Method (`lstm.py`) |
|---|---|---|
| Mean μ | Linear regression on rolling-average features | LSTM over the player's last 10 games + tonight's context |
| Spread σ | One fixed value for every player (std of training errors) | Predicted per player-game (softplus head) |
| Training | Least squares | Gaussian negative log-likelihood (NLL), Adam, early stopping on validation NLL |

### Metrics

| Metric | Measures | Better |
|---|---|---|
| **NLL** | Quality of the whole predicted distribution (μ and σ). Main metric. | Lower |
| **ECE** | Calibration: when we say "70% over", does it go over ~70% of the time? | Lower |
| **MAE** | Average miss of μ, in points | Lower |
| **Hit rate** | % of correct over/under picks against the line | Higher |

> **Note:** until real sportsbook lines are logged, the line is a **proxy**: the player's 10-game scoring average rounded to the nearest 0.5. Hit rate against the proxy is not a betting result.

---

## Project structure

```
NothingButNet/
├── run_pipeline.py    # runs steps 1-4 + EDA with one command   
├── setup_data.py      # 1. download raw NBA data (nba_api) -> data/raw/
├── build_teams.py     # 2. clean + build features          -> data/clean/
├── baseline.py        # 3. linear regression + fixed sigma -> runs/, models/
├── lstm.py            # 4. LSTM with mean + sigma head     -> runs/, models/
├── eda.py             # data exploration plots              -> figures/
├── log_lines.py       # daily logger for real prop lines    -> data/lines/
├── runlog.py          # helper: one folder per run (params, log, metrics, plots)
├── figures/           # EDA plots
├── split.py           # ONE shared definition of the split, features and row filter
├── requirements.txt
├── data/              # created by the scripts, NOT committed
│   ├── raw/           #   player_gamelogs/, team_gamelogs/, schedule/
│   └── clean/         #   players_features.parquet, team_features.parquet
├── models/            # latest trained weights, NOT committed
└── runs/              # one folder per training run (weights inside are NOT committed)
```

---

## Setup

Requires **Python 3.12**.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (with an NVIDIA GPU)

Install the CUDA build of PyTorch **first**, using the command from the selector at <https://pytorch.org/get-started/locally/>. Otherwise pip installs the CPU-only build. Then:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # example: use the command pytorch.org gives you
pip install -r requirements.txt
```

Check the GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

### VS Code

Open the `NothingButNet` folder (not a single file). Press **Cmd/Ctrl+Shift+P → Python: Select Interpreter** and choose `.venv` (marked *Workspace*). Then open a **new** terminal; the prompt should start with `(.venv)`.

---

## How to run

Run every command **from the project root**.

### Everything with one command

```bash
python run_pipeline.py              # download (if needed) -> features -> EDA -> baseline -> LSTM
python run_pipeline.py --frac 0.1   # quick check on 10% of players (~10 s once data exists)
```

| Flag | Effect |
|---|---|
| `--frac 0.1` | Train and evaluate the models on a random 10% of **players** (all their games). Same seed = same players. |
| `--seed 7` | Seed for weights, shuffling and player sampling (default 42) |
| `--epochs 5` | Maximum LSTM epochs (early stopping can end sooner) |
| `--download` | Re-download the raw data even if `data/raw/` exists (first run downloads automatically) |
| `--skip-eda` | Skip the data exploration plots |

The pipeline stops at the first step that fails. `baseline.py` and `lstm.py` accept the same `--frac` and `--seed` flags (and `lstm.py` also `--epochs`) when run on their own.

### Or step by step

#### 1. Download the data (once)

```bash
python setup_data.py
```

Downloads player game logs, team game logs and the league schedule for every season from 2016-17 to the current one. It takes a few minutes and pauses between requests so NBA.com doesn't throttle it. Run it on your own machine: NBA.com often blocks cloud servers such as Colab. It is safe to re-run: finished seasons are skipped and only the current season is refreshed.

#### 2. Build the features

```bash
python build_teams.py
```

Produces `data/clean/players_features.parquet` (one row per player per game). It contains:

- **Player form:** rolling 5- and 10-game averages of points, minutes, field-goal attempts and free-throw attempts, plus the 10-game points standard deviation
- **Context:** home/away, rest days, back-to-back flag, games played
- **Opponent:** rolling 10-game points allowed, pace and defensive rating
- **Proxy line:** 10-game points average, rounded to 0.5

All rolling features use **previous games only** (`shift(1)`), so a row never contains information from the game being predicted.

#### 3. Train and evaluate the baseline

```bash
python baseline.py
```

#### 4. Train and evaluate the LSTM

```bash
python lstm.py
```

Run the baseline first: `lstm.py` loads the saved baseline to print a side-by-side comparison and to draw it on the learning curve. On CPU the LSTM trains in about a minute; on a GPU it is faster.

### Data exploration

```bash
python eda.py
```

Saves six plots and a text summary to `figures/`:

| File | Shows |
|---|---|
| `eda_1_rows_per_season.png` | Player-games per season and per split, and how many survive the missing-feature filter |
| `eda_2_points_distribution.png` | Distribution of points vs. a Gaussian with the same mean and std |
| `eda_3_spread_by_scoring_level.png` | Spread of (points − 10-game average) by scoring level, vs. one fixed σ |
| `eda_4_over_rate_vs_proxy_line.png` | How often players go over the proxy line, by scoring level |
| `eda_5_context_effects.png` | Home/away and rest-day effects on scoring vs. recent form |
| `eda_6_missing_values.png` | Missing values per feature |

All plots except the first use **training seasons only**, so validation and test data never influence modelling choices.

### Logging real prop lines (daily)

The free plan of [The Odds API](https://the-odds-api.com) has no historical odds, so real lines have to be saved **before each game, every day**.

1. Get a free API key (500 credits/month) at <https://the-odds-api.com>.
2. Create a file named `.env` in the project root containing `ODDS_API_KEY=your_key_here`. It is git-ignored: **never commit the key**.
3. Once a day, a few hours before the first tip-off:

```bash
python log_lines.py --dry-run   # list today's games, uses no credits
python log_lines.py             # download player-points lines (~1 credit per game)
```

Each run saves the raw API responses to `data/lines/raw/` and appends one row per player × bookmaker × Over/Under to `data/lines/player_points_lines.csv`. One run per day fits in the free credits; the script prints how many are left. Like the rest of `data/`, these files are not committed, so whoever runs the logger should back them up.

### Data split (by season, no look-ahead)

| Split | Seasons | Used for |
|---|---|---|
| Train | 2016-17 → 2023-24 | Fitting |
| Validation | 2024-25 | Early stopping, model selection, all reported comparisons |
| Test | 2025-26 onward | Evaluated **once**, at the end of the project |

---

## Outputs: run folders

Every run of `baseline.py` or `lstm.py` creates its own folder, so each reported number traces back to the run that produced it:

```
runs/2026-11-20_18-06-40_lstm/
├── params.json          # settings, seed, git commit, library versions, command
├── log.txt              # everything the script printed
├── metrics.json         # final validation metrics (overall + by scorer tier)
├── history.csv          # LSTM only: train/val NLL per epoch
├── learning_curve.png   # LSTM only: train vs val NLL, best epoch, baseline level
└── best.pt              # LSTM only: best weights (not committed)
```

`params.json` records the git commit. A commit ending in `-dirty` means the code had uncommitted changes when it ran, so commit first, then run, for results you plan to report.

`models/` always holds the **latest** baseline and LSTM weights.

---

## Current status

- [x] Data download and feature pipeline on real NBA data
- [x] Linear-regression baseline with fixed σ
- [x] LSTM with per-game σ, trained with Gaussian NLL and early stopping
- [x] Run folders and learning curve
- [x] Data exploration plots (`eda.py`)
- [x] Daily logger for real prop lines (`log_lines.py`, The Odds API)
- [ ] Logger running every day, plus matching logged lines to box scores (player names → `PLAYER_ID`)
- [ ] Injury "temperature" on σ
- [ ] News "temperature" on σ (GDELT headlines + DistilBERT)
- [ ] Controlled experiment: base vs. +injuries vs. +news
- [ ] Final evaluation on the test seasons
- [ ] Live demo / inference script

---

## Data and responsible use

- Box scores and schedules come from NBA.com through the open-source [`nba_api`](https://github.com/swar/nba_api) package. Data is **not** committed to this repo; run `setup_data.py` to reproduce it.
- Prop lines come from [The Odds API](https://the-odds-api.com) under its free plan terms. They are not committed either.
- Never commit datasets, large weight files or API keys.
- This is a course project about uncertainty estimation. It is **not** betting advice, and the model must not be used to make decisions about real people.

---

## References

- S. Patel, `nba_api`: an API client for NBA.com. <https://github.com/swar/nba_api>
- S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 1997.
- C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On calibration of modern neural networks," *ICML*, 2017.
- B. Lakshminarayanan, A. Pritzel and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," *NeurIPS*, 2017.
- V. Sanh, L. Debut, J. Chaumond and T. Wolf, "DistilBERT, a distilled version of BERT," 2019.
