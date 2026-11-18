import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
out = ROOT / "data" / "clean" / "player_games.parquet"
out.parent.mkdir(parents=True, exist_ok=True)

player_dir = Path("./data/raw/player_gamelogs")
schedule_dir = Path("./data/raw/schedule")
team_dir = Path("./data/raw/team_gamelogs")


players = pd.read_parquet(player_dir)
players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])

teams = pd.read_parquet(team_dir)
pd.set_option("display.max_columns", None)
teams["GAME_DATE"] = pd.to_datetime(teams["GAME_DATE"])
n_before = len(teams)

# 1. Possessions per team per game (0.44 standard estimate coming from pro basketball analysts)
teams["POSS"] = teams["FGA"] + 0.44 * teams["FTA"] - teams["OREB"] + teams["TOV"]

#Attach the OTHER team's numbers from the same game (Create columns for opp and colums for main team [this creates duplicate rows], filter our rows that have the same team for both stats)
opp = teams[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "PTS", "POSS"]].rename(columns={
    "TEAM_ID": "OPP_TEAM_ID", "TEAM_ABBREVIATION": "OPP_TEAM_ABBREVIATION", "TEAM_NAME": "OPP_TEAM_NAME", "PTS": "PTS_ALLOWED", "POSS": "OPP_POSS"})
teams = teams.merge(opp, on="GAME_ID")
teams = teams[teams["TEAM_ID"] != teams["OPP_TEAM_ID"]]
assert len(teams) == n_before, "each game should have exactly 2 team rows"

#3. Per-game defensive / pace stats
teams["PACE"]    = (teams["POSS"] + teams["OPP_POSS"]) / 2
teams["DEF_RTG"] = 100 * teams["PTS_ALLOWED"] / teams["OPP_POSS"]  # pts allowed per 100 poss

# --- 4. Rolling averages over PREVIOUS games only
teams = teams.sort_values(["TEAM_ID", "GAME_DATE"])
g = teams.groupby(["TEAM_ID", "SEASON"])
for col in ["PTS_ALLOWED", "PACE", "DEF_RTG"]:
    teams[f"{col}_ROLL10"] = g[col].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())

teams.to_parquet("data/clean/team_features.parquet", index=False)
print(teams.columns.tolist())
print(teams[["TEAM_ABBREVIATION", "OPP_TEAM_ABBREVIATION", "OPP_TEAM_NAME"]].head())

n_players = len(players)

# --- Join 1: who is the opponent?
opp_info = teams[["GAME_ID", "TEAM_ID", "OPP_TEAM_ID",
                  "OPP_TEAM_ABBREVIATION", "OPP_TEAM_NAME"]]
players = players.merge(opp_info, on=["GAME_ID", "TEAM_ID"],
                        how="left", validate="many_to_one")

# --- Join 2: that opponent's defense coming into the game
opp_feats = teams[["GAME_ID", "TEAM_ID",
                   "PTS_ALLOWED_ROLL10", "PACE_ROLL10", "DEF_RTG_ROLL10"]].rename(columns={
    "TEAM_ID": "OPP_TEAM_ID",
    "PTS_ALLOWED_ROLL10": "OPP_PTS_ALLOWED_ROLL10",
    "PACE_ROLL10": "OPP_PACE_ROLL10",
    "DEF_RTG_ROLL10": "OPP_DEF_RTG_ROLL10",
})
players = players.merge(opp_feats, on=["GAME_ID", "OPP_TEAM_ID"],
                        how="left", validate="many_to_one")

assert len(players) == n_players, "join duplicated rows!"
print(players[["OPP_TEAM_ID", "OPP_DEF_RTG_ROLL10"]].isna().sum())

# --- 1. Home / away from MATCHUP ("LAL vs. BOS" = home, "LAL @ BOS" = away)
players["HOME"] = players["MATCHUP"].str.contains("vs.", regex=False).astype(int)

# --- 2. Tip-off time from the schedule (for the news cutoff later)
schedule = pd.read_parquet(schedule_dir)[["gameId", "gameDateTimeUTC"]]
schedule = schedule.rename(columns={"gameId": "GAME_ID", "gameDateTimeUTC": "TIPOFF_UTC"})
schedule["TIPOFF_UTC"] = pd.to_datetime(schedule["TIPOFF_UTC"], utc=True)
players = players.merge(schedule, on="GAME_ID", how="left", validate="many_to_one")
assert len(players) == n_players

# --- 3. Sort each player's games in date order (required for everything below)
players = players.sort_values(["PLAYER_ID", "GAME_DATE"])
gp = players.groupby("PLAYER_ID")

# --- 4. Rest days and back-to-backs
players["REST_DAYS"] = gp["GAME_DATE"].diff().dt.days.clip(upper=7)
players["B2B"] = (players["REST_DAYS"] == 1).astype(int)

# --- 5. Player rolling features (previous games only!)
for col in ["PTS", "MIN", "FGA", "FTA"]:
    for w in [5, 10]:
        players[f"{col}_ROLL{w}"] = gp[col].transform(
            lambda s: s.shift(1).rolling(w, min_periods=3).mean())
players["PTS_STD10"] = gp["PTS"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=3).std())

# --- 6. Proxy prop line: recent scoring average rounded to nearest 0.5
players["PROXY_LINE"] = (players["PTS_ROLL10"] * 2).round() / 2

# --- 7. Save
players.to_parquet("data/clean/players_features.parquet")

#for parquet_path in sorted(player_dir.glob("*.parquet")): #glob() returns all files matching the pattern, sorted() sorts them alphabetically
    #print(f"Shape: {df.shape}")
    #print(f"Data types:\n{df.dtypes}")    
    #print(f"GATE_DATE:\n{df["GAME_DATE"].dtype}")
    #print(f"DF HEAD:\n{df.head()}")
    #print(f"Missing:{df.isna().sum()}")        # how many rows have no value for that column
    #print(f"Description:\n{df.describe()}")          # gives quick statistical summary of each numeric column, which spots weird values

