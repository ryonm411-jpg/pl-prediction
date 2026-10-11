import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import build_db  # noqa: E402

OLD_SEASON = """\
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A
E0,13/08/05,Aston Villa,Bolton,2,2,D,3,13,2,5,7,8,0,2,0,0,2.3,3.25,3
E0,14/08/05,Man City,West Brom,0,0,D,15,6,8,3,9,2,0,0,0,0,1.72,3.4,5
,,,,,,,,,,,,,,,,,,,
"""

NEW_SEASON = """\
Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA
E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,19,10,10,3,6,7,1,2,0,0,1.3,6,9.5,1.29,6.2,10.5,1.28,6,10
E0,16/08/2025,12:30,Aston Villa,Newcastle,0,0,D,3,16,1,3,3,9,1,3,1,0,2.3,3.4,3.1,,,,2.4,3.5,2.9
"""


@pytest.fixture
def db(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # Mirrors the real files: old seasons have two-digit years and trailing blank
    # rows; recent seasons have four-digit years and a UTF-8 byte-order mark.
    (raw / "E0_2005-06.csv").write_text(OLD_SEASON, encoding="utf-8")
    (raw / "E0_2025-26.csv").write_text(NEW_SEASON, encoding="utf-8-sig")
    db_path = tmp_path / "test.db"
    n = build_db.build(raw_dir=raw, db_path=db_path)
    conn = sqlite3.connect(db_path)
    yield n, conn
    conn.close()


def test_loads_every_played_match_and_skips_blank_rows(db):
    n, conn = db
    assert n == 4
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 4


def test_two_and_four_digit_years_become_iso_dates(db):
    _, conn = db
    dates = [r[0] for r in conn.execute("SELECT match_date FROM matches ORDER BY match_date")]
    assert dates == ["2005-08-13", "2005-08-14", "2025-08-15", "2025-08-16"]


def test_season_comes_from_file_name(db):
    _, conn = db
    seasons = conn.execute("SELECT season, COUNT(*) FROM matches GROUP BY season").fetchall()
    assert seasons == [("2005-06", 2), ("2025-26", 2)]


def test_columns_absent_from_old_seasons_are_null(db):
    _, conn = db
    row = conn.execute(
        "SELECT kickoff_time, pinnacle_close_home, market_avg_close_home, b365_home "
        "FROM matches WHERE season = '2005-06' AND home_team = 'Aston Villa'"
    ).fetchone()
    assert row == (None, None, None, 2.3)


def test_missing_pinnacle_odds_stay_null_while_market_average_is_kept(db):
    _, conn = db
    row = conn.execute(
        "SELECT pinnacle_close_home, market_avg_close_home FROM matches WHERE home_team = 'Aston Villa' AND season = '2025-26'"
    ).fetchone()
    assert row == (None, 2.4)


def test_rebuild_replaces_rather_than_duplicates(tmp_path, db):
    _, conn = db
    db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    build_db.build(raw_dir=tmp_path / "raw", db_path=db_path)
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 4
