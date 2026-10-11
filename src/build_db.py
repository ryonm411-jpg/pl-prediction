"""Load the season CSVs in data/raw/ into the SQLite database data/epl.db.

The database is rebuilt from scratch on every run, so it always matches the
CSVs exactly. Run src/fetch_data.py first.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "epl.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# football-data.co.uk column -> matches table column
COLUMNS = {
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    "B365H": "b365_home",
    "B365D": "b365_draw",
    "B365A": "b365_away",
    "PSCH": "pinnacle_close_home",
    "PSCD": "pinnacle_close_draw",
    "PSCA": "pinnacle_close_away",
    "AvgCH": "market_avg_close_home",
    "AvgCD": "market_avg_close_draw",
    "AvgCA": "market_avg_close_away",
}


def load_season_csv(path: Path) -> pd.DataFrame:
    """Read one season file into rows shaped like the matches table."""
    season = path.stem.removeprefix("E0_")
    # Some seasons start with a UTF-8 byte-order mark; blank trailing rows have no HomeTeam.
    raw = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["HomeTeam"])

    df = pd.DataFrame({"season": season}, index=raw.index)
    # Dates are dd/mm/yy in some seasons and dd/mm/yyyy in others.
    df["match_date"] = pd.to_datetime(raw["Date"], format="mixed", dayfirst=True).dt.strftime("%Y-%m-%d")
    df["kickoff_time"] = raw["Time"] if "Time" in raw.columns else None
    for src, dst in COLUMNS.items():
        df[dst] = raw[src] if src in raw.columns else None
    return df


def build(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> int:
    files = sorted(raw_dir.glob("E0_*.csv"))
    if not files:
        raise FileNotFoundError(f"No season CSVs in {raw_dir}; run src/fetch_data.py first")

    matches = pd.concat([load_season_csv(f) for f in files], ignore_index=True)
    matches = matches.sort_values(["match_date", "kickoff_time", "home_team"], na_position="first")

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        matches.to_sql("matches", conn, if_exists="append", index=False)
    return len(matches)


def main() -> None:
    n = build()
    print(f"Loaded {n} matches into {DB_PATH}")


if __name__ == "__main__":
    sys.exit(main())
