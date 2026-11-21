"""
log_lines.py - Daily logger for real NBA player-points prop lines (The Odds API).

Why: the free API plan has no historical odds, so the only way to get real
lines for evaluation is to save them ourselves, every day, BEFORE tip-off.
Every day this doesn't run is lost test data.

What it does (one run = one snapshot):
  1. Lists NBA games starting in the next HOURS_AHEAD hours   (free, no credits)
  2. For each game, downloads the 'player_points' over/under market
     from US bookmakers                                         (1 credit per game)
  3. Saves the raw JSON (data/lines/raw/) and appends one row per
     player x bookmaker x side to data/lines/player_points_lines.csv

Setup (once):
  * Get a free key at https://the-odds-api.com  (500 credits/month)
  * Put it in a file named .env in the project root:   ODDS_API_KEY=your_key_here
    (.env is in .gitignore; NEVER commit the key)

Run from the project root, once a day, a few hours before the first game:
    python log_lines.py              # log today's games
    python log_lines.py --dry-run    # only list the games, uses no credits

Budget: ~1 credit per game per run. The NBA averages roughly 8 games a night
(max 15), so ONE run per day fits in the free 500 credits/month. Running twice a
day would not. Use --max-games to cap a run. Credits left are printed every run.
"""

import argparse
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "lines"
RAW_DIR = OUT_DIR / "raw"
CSV_PATH = OUT_DIR / "player_points_lines.csv"

API = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
MARKET = "player_points"
REGIONS = "us"
HOURS_AHEAD = 24          # only games tipping off within this window
TIMEOUT = 30

CSV_COLUMNS = [
    "snapshot_utc",       # when WE downloaded it
    "event_id", "commence_time", "home_team", "away_team",
    "bookmaker", "market_last_update",
    "player_name", "side", "line", "price_american",
]


# ------------------------------------------------------------------ helpers
def load_api_key():
    """Read ODDS_API_KEY from the environment, or from a .env file in the project root."""
    key = os.environ.get("ODDS_API_KEY")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("ODDS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("No API key. Create .env with ODDS_API_KEY=... (see top of log_lines.py)")
    return key


def get(path, params):
    resp = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
    if resp.status_code != 200:
        # never print the full URL: it contains the API key
        raise RuntimeError(f"{path} -> HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json(), resp.headers


def iso(dt):
    """The API wants times like 2026-11-20T23:00:00Z (no microseconds)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def flatten(event, snapshot_utc):
    """Turn one event-odds response into CSV rows: one per player x bookmaker x Over/Under."""
    rows = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != MARKET:
                continue
            for out in market.get("outcomes", []):
                rows.append({
                    "snapshot_utc": snapshot_utc,
                    "event_id": event["id"],
                    "commence_time": event["commence_time"],
                    "home_team": event["home_team"],
                    "away_team": event["away_team"],
                    "bookmaker": book["key"],
                    "market_last_update": market.get("last_update", ""),
                    # for player props: name = "Over"/"Under", description = player
                    "player_name": out.get("description", ""),
                    "side": out.get("name", ""),
                    "line": out.get("point", ""),
                    "price_american": out.get("price", ""),
                })
    return rows


def append_rows(rows):
    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description="Log today's NBA player-points prop lines.")
    parser.add_argument("--dry-run", action="store_true",
                        help="only list upcoming games (free), don't download odds")
    parser.add_argument("--max-games", type=int, default=None,
                        help="download at most this many games (to save credits)")
    args = parser.parse_args()

    key = load_api_key()
    now = datetime.now(timezone.utc)
    snapshot_utc = iso(now)

    # 1. Upcoming games (this endpoint costs no credits)
    events, _ = get(f"/sports/{SPORT}/events", {
        "apiKey": key,
        "commenceTimeFrom": iso(now),                      # not started yet -> pre-game lines
        "commenceTimeTo": iso(now + timedelta(hours=HOURS_AHEAD)),
    })
    print(f"{len(events)} NBA games in the next {HOURS_AHEAD} h:")
    for e in events:
        print(f"  {e['commence_time']}  {e['away_team']} @ {e['home_team']}")
    if args.dry_run or not events:
        return
    if args.max_games is not None:
        events = events[:args.max_games]

    # 2-3. Odds per game, saved raw + flattened
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total_rows, remaining = 0, "?"
    for e in events:
        try:
            data, headers = get(f"/sports/{SPORT}/events/{e['id']}/odds", {
                "apiKey": key, "regions": REGIONS, "markets": MARKET,
                "oddsFormat": "american", "dateFormat": "iso",
            })
        except Exception as err:
            print(f"  ! {e['away_team']} @ {e['home_team']}: {err}")
            continue
        remaining = headers.get("x-requests-remaining", remaining)

        raw_path = RAW_DIR / f"{now:%Y-%m-%d_%H%M}_{e['id']}.json"
        raw_path.write_text(json.dumps(data, indent=1))   # keep the raw copy in case we
                                                          # need fields we didn't flatten
        rows = flatten(data, snapshot_utc)
        append_rows(rows)
        total_rows += len(rows)
        n_players = len({r["player_name"] for r in rows})
        print(f"  {e['away_team']} @ {e['home_team']}: {n_players} players, {len(rows)} rows "
              f"(cost {headers.get('x-requests-last', '?')})")

    print(f"\nappended {total_rows} rows to {CSV_PATH.relative_to(ROOT)}")
    print(f"credits remaining this month: {remaining}")


if __name__ == "__main__":
    main()
