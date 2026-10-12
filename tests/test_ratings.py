import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ratings import INITIAL_ELO, EloTable, expected_home_score  # noqa: E402


def test_equal_teams_favour_the_home_side():
    assert expected_home_score(1500, 1500) > 0.5
    assert expected_home_score(1500, 1500, home_advantage=0) == 0.5


def test_home_win_moves_points_from_away_to_home():
    elo = EloTable()
    elo.start_season({"Arsenal", "Chelsea"})
    elo.update("Arsenal", "Chelsea", "H")
    gain = elo["Arsenal"] - INITIAL_ELO
    assert gain > 0
    assert elo["Chelsea"] == pytest.approx(INITIAL_ELO - gain)


def test_home_draw_against_equal_team_loses_points():
    # The home side was expected to score more than 0.5, so a draw disappoints.
    elo = EloTable()
    elo.start_season({"Arsenal", "Chelsea"})
    elo.update("Arsenal", "Chelsea", "D")
    assert elo["Arsenal"] < INITIAL_ELO


def test_total_elo_is_constant_within_a_season():
    elo = EloTable()
    teams = {"A", "B", "C", "D"}
    elo.start_season(teams)
    for home, away, result in [("A", "B", "H"), ("C", "D", "A"), ("B", "C", "D"), ("D", "A", "H")]:
        elo.update(home, away, result)
    assert sum(elo[t] for t in teams) == pytest.approx(4 * INITIAL_ELO)


def test_promoted_team_inherits_relegated_average_then_everyone_regresses():
    elo = EloTable(season_regression=0.2)
    elo.start_season({"A", "B", "C"})
    elo.ratings.update({"A": 1600.0, "B": 1500.0, "C": 1400.0})

    elo.start_season({"A", "B", "Promoted"})  # C relegated

    assert elo["A"] == pytest.approx(1600 - 0.2 * 100)
    assert elo["Promoted"] == pytest.approx(1400 + 0.2 * 100)
    assert sum(elo[t] for t in ("A", "B", "Promoted")) == pytest.approx(3 * INITIAL_ELO)
