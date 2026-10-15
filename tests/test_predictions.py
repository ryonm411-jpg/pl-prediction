import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import predictions  # noqa: E402

SNAPSHOT = {"id": "doku", "reason": "pace", "favours": "Man City", "elo": 25}


@pytest.fixture
def dbs(tmp_path):
    matches_db = tmp_path / "epl.db"
    with sqlite3.connect(matches_db) as conn:
        conn.execute("CREATE TABLE matches (season, match_date, home_team, away_team, result, home_goals, away_goals)")
        conn.execute("INSERT INTO matches VALUES ('2026-27', '2026-10-18', 'Man City', 'Liverpool', 'H', 2, 0)")
    conn = predictions.connect(tmp_path / "predictions.db")
    yield conn, matches_db
    conn.close()


def test_logged_prediction_is_scored_once_the_result_exists(dbs):
    conn, matches_db = dbs
    predictions.log(conn, "2026-27", "Man City", "Liverpool", np.array([0.2, 0.3, 0.5]),
                    np.array([0.15, 0.25, 0.6]), None, [SNAPSHOT])
    predictions.log(conn, "2026-27", "Arsenal", "Chelsea", np.array([0.3, 0.3, 0.4]),
                    np.array([0.3, 0.3, 0.4]), None, [])

    df = predictions.scored(conn, matches_db)
    assert list(df["home_team"]) == ["Man City"]  # Arsenal v Chelsea not played yet

    overall, per_adjustment = predictions.report(df)
    lls = overall.set_index("forecast")["log_loss"]
    assert lls["Model"] == pytest.approx(-np.log(0.5))
    assert lls["Model + Analyst Adjustments"] == pytest.approx(-np.log(0.6))
    record = per_adjustment.iloc[0]
    assert (record["adjustment"], record["applied"], record["helped"]) == ("doku", 1, 1)


def test_re_predicting_before_kickoff_replaces_the_earlier_prediction(dbs):
    conn, _ = dbs
    for p_home in (0.5, 0.55):
        p = np.array([0.25, 0.2, p_home])
        predictions.log(conn, "2026-27", "Man City", "Liverpool", p, p, None, [])
    rows = conn.execute("SELECT model_home FROM predictions").fetchall()
    assert rows == [(0.55,)]


def test_adjustment_snapshot_is_stored_verbatim(dbs):
    conn, _ = dbs
    p = np.array([0.2, 0.3, 0.5])
    predictions.log(conn, "2026-27", "Man City", "Liverpool", p, p, None, [SNAPSHOT])
    stored = conn.execute("SELECT adjustments FROM predictions").fetchone()[0]
    assert json.loads(stored) == [SNAPSHOT]
