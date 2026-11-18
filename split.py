
import pandas as pd
from pathlib import Path

#settings
ROOT = Path(__file__).resolve().parent
out = ROOT / "data" / "clean" / "player_games.parquet"
out.parent.mkdir(parents=True, exist_ok=True)

player_dir = Path("./data/clean/players_features.parquet")
team_dir = Path("./data/clean/team_features.parquet")

players = pd.read_parquet(player_dir)



train = players[players["SEASON"] <= "2023-24"]
val   = players[players["SEASON"] == "2024-25"]
test  = players[players["SEASON"] >= "2025-26"] 

FEATURES = [
    "PTS_ROLL5", "PTS_ROLL10", "PTS_STD10",
    "MIN_ROLL5", "MIN_ROLL10",
    "FGA_ROLL5", "FGA_ROLL10", "FTA_ROLL5", "FTA_ROLL10",
    "REST_DAYS", "B2B", "HOME",
    "OPP_PTS_ALLOWED_ROLL10", "OPP_PACE_ROLL10", "OPP_DEF_RTG_ROLL10",
]
#Target we are trying to predict
TARGET = "PTS"

#Drop rows that have empty values for the above columns
train = train.dropna(subset=FEATURES + [TARGET]).copy()
val   = val.dropna(subset=FEATURES + [TARGET]).copy()
test  = test.dropna(subset=FEATURES + [TARGET]).copy()

print(train)
print(val)
print(test)