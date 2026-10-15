"""Prediction log: every forecast made before Kickoff, scored once the Result is in.

Stored in data/predictions.db, separate from data/epl.db: epl.db is rebuilt from
the CSVs at will, but Predictions cannot be recreated, so this file is committed.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import score

PREDICTIONS_DB = Path(__file__).resolve().parent.parent / "data" / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   INTEGER PRIMARY KEY,
    created_at      TEXT NOT NULL,              -- UTC, ISO 8601
    season          TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    model_away      REAL NOT NULL,
    model_draw      REAL NOT NULL,
    model_home      REAL NOT NULL,
    adjusted_away   REAL NOT NULL,
    adjusted_draw   REAL NOT NULL,
    adjusted_home   REAL NOT NULL,
    market_away     REAL,                       -- from odds entered by hand, if any
    market_draw     REAL,
    market_home     REAL,
    adjustments     TEXT NOT NULL,              -- JSON list of Analyst Adjustment snapshots
    UNIQUE (season, home_team, away_team)       -- re-predicting before Kickoff replaces
);
"""

PROBS = ["away", "draw", "home"]


def connect(path: Path = PREDICTIONS_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def log(conn: sqlite3.Connection, season: str, home_team: str, away_team: str, model: np.ndarray,
        adjusted: np.ndarray, market: np.ndarray | None, adjustments: list[dict]) -> None:
    market = market if market is not None else [None] * 3
    conn.execute(
        "INSERT OR REPLACE INTO predictions (created_at, season, home_team, away_team, "
        "model_away, model_draw, model_home, adjusted_away, adjusted_draw, adjusted_home, "
        "market_away, market_draw, market_home, adjustments) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), season, home_team, away_team,
         *map(float, model), *map(float, adjusted), *market, json.dumps(adjustments)),
    )
    conn.commit()


def scored(conn: sqlite3.Connection, matches_db: Path) -> pd.DataFrame:
    """Predictions whose Match has been played, joined to the Result."""
    conn.execute("ATTACH DATABASE ? AS epl", (str(matches_db),))
    try:
        return pd.read_sql("""
            SELECT p.*, m.match_date, m.result, m.home_goals, m.away_goals
            FROM predictions p
            JOIN epl.matches m USING (season, home_team, away_team)
            ORDER BY m.match_date
        """, conn)
    finally:
        conn.execute("DETACH DATABASE epl")


def report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overall scores (model, adjusted, market) and a record per Analyst Adjustment."""
    def probs(prefix: str, rows: pd.DataFrame) -> np.ndarray:
        return rows[[f"{prefix}_{p}" for p in PROBS]].to_numpy(dtype=float)

    overall = [{"forecast": "Model", "matches": len(df), **score(df["result"], probs("model", df))},
               {"forecast": "Model + Analyst Adjustments", "matches": len(df),
                **score(df["result"], probs("adjusted", df))}]
    with_market = df.dropna(subset=["market_home"])
    if len(with_market):
        overall.append({"forecast": "Market (matches with odds)", "matches": len(with_market),
                        **score(with_market["result"], probs("market", with_market))})

    per_adjustment: dict[str, dict] = {}
    for _, row in df.iterrows():
        applied = json.loads(row["adjustments"])
        if not applied:
            continue
        one = row.to_frame().T
        model_ll = score(one["result"], probs("model", one))["log_loss"]
        adjusted_ll = score(one["result"], probs("adjusted", one))["log_loss"]
        for a in applied:
            rec = per_adjustment.setdefault(a["id"], {"adjustment": a["id"], "applied": 0, "helped": 0,
                                                      "log_loss_change": 0.0})
            rec["applied"] += 1
            rec["helped"] += int(adjusted_ll < model_ll)
            rec["log_loss_change"] += adjusted_ll - model_ll
    return pd.DataFrame(overall), pd.DataFrame(per_adjustment.values())
