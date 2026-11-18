"""
setup_data.py - Download the raw NBA datasets for Nothing But Net. To only be ran once (Or as many times as it takes  to grab every seasons' data).

Fetches, for each season:
  1. Player game logs  (one row per player per game)  -> target + player features
  2. Team game logs    (one row per team per game)    -> opponent features
  3. League schedule   (one row per game, with tip-off times in UTC) -> news cutoff

Run from the project root:
    python setup_data.py

Safe to re-run: finished seasons are skipped; only the current season is refreshed.
Run this on your own machine; NBA.com often blocks cloud servers such as Colab.
"""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, scheduleleaguev2

# SET UP SETTINGS
RAW_DIR = Path("data/raw")

PAST_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2026)]  # 2016-17 ... 2025-26
CURRENT_SEASON = "2026-27"  # re-downloaded every run, since new games keep being added
SEASON_TYPES = ["Regular Season", "Playoffs"]

PAUSE_SECONDS = 1.5  # wait between requests so NBA.com doesn't throttle us
MAX_RETRIES = 3
TIMEOUT = 60


# Helps limit fetching rate
def fetch_with_retry(make_request, label):
    """Call make_request(); if it fails, wait longer each time and try again."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return make_request()
        except Exception as err:
            wait = PAUSE_SECONDS * 4 * attempt
            print(f"   ! {label} failed (attempt {attempt}/{MAX_RETRIES}): {err}")
            if attempt < MAX_RETRIES:
                print(f"     retrying in {wait:.0f}s...")
                time.sleep(wait)
    print(f"   x giving up on {label}; re-run the script later to try again")
    return None


def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"   saved {path}  ({len(df):,} rows, {df.shape[1]} cols)")


def needs_download(path, season):
    """Download if the file is missing, or if it's the season still being played."""
    return season == CURRENT_SEASON or not path.exists()


# Game Fetchers
def fetch_game_logs(season, season_type, who):
    """who = 'P' for players, 'T' for teams."""
    kind = "player" if who == "P" else "team"
    tag = season_type.lower().replace(" ", "_")  # 'regular_season' / 'playoffs'
    path = RAW_DIR / f"{kind}_gamelogs" / f"{kind}_{season}_{tag}.parquet"

    if not needs_download(path, season):
        print(f"   skip {path.name} (already downloaded)")
        return

    def request():
        resp = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation=who,
            timeout=TIMEOUT,
        )
        return resp.get_data_frames()[0]

    df = fetch_with_retry(request, f"{kind} {season} {season_type}")
    time.sleep(PAUSE_SECONDS)
    if df is None:
        return
    if df.empty:
        print(f"   (no {season_type} games for {season} yet)")
        return

    df["GAME_ID"] = df["GAME_ID"].astype(str)  # keep leading zeros: '0022400001'
    df["SEASON"] = season                       # game log has no readable season column
    df["SEASON_TYPE"] = season_type             # ...or season type
    save(df, path)


def fetch_schedule(season):
    path = RAW_DIR / "schedule" / f"schedule_{season}.parquet"

    if not needs_download(path, season):
        print(f"   skip {path.name} (already downloaded)")
        return

    def request():
        resp = scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=TIMEOUT)
        return resp.get_data_frames()[0]  # 'SeasonGames' table

    df = fetch_with_retry(request, f"schedule {season}")
    time.sleep(PAUSE_SECONDS)
    if df is None or df.empty:
        print(f"   (no schedule returned for {season}; older seasons may not be "
              f"available from this endpoint; fall back to the CSV's gameDateTimeEst)")
        return

    df["gameId"] = df["gameId"].astype(str)
    save(df, path)


# Main
def main():
    seasons = PAST_SEASONS + [CURRENT_SEASON]
    print(f"Downloading {len(seasons)} seasons into {RAW_DIR}/\n")

    for season in seasons:
        print(f"== {season} ==")
        for season_type in SEASON_TYPES:
            fetch_game_logs(season, season_type, who="P")
            fetch_game_logs(season, season_type, who="T")
        fetch_schedule(season)
        print()

    print("Done. Load everything later with, e.g.:")
    print('   pd.read_parquet("data/raw/player_gamelogs")')


if __name__ == "__main__":
    main()
