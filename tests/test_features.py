import sys
from itertools import permutations
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_dataset import FEATURE_COLUMNS, STATS_WINDOW, compute_features  # noqa: E402


def make_league() -> pd.DataFrame:
    """Two seasons of a 4-team league; D is relegated and E promoted in between.

    Results and scores are a deterministic function of the fixture so every
    team ends up with a different history. One Match is played per date.
    """
    rows = []
    match_id = 0
    for season, teams, year in [("2005-06", "ABCD", 2005), ("2006-07", "ABCE", 2006)]:
        fixtures = list(permutations(teams, 2))
        for i, (home, away) in enumerate(fixtures):
            home_goals = (ord(home) + i) % 4
            away_goals = (ord(away) + 2 * i) % 3
            result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
            rows.append({
                "match_id": match_id,
                "season": season,
                "match_date": f"{year}-09-{1 + i:02d}",
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "home_shots_on_target": home_goals + 2,
                "away_shots_on_target": away_goals + 1,
            })
            match_id += 1
    return pd.DataFrame(rows)


def test_one_feature_row_per_match():
    league = make_league()
    features = compute_features(league)
    assert sorted(features["match_id"]) == sorted(league["match_id"])


@pytest.mark.parametrize("match_id", [3, 12, 20])
def test_changing_a_result_never_changes_features_on_or_before_its_date(match_id):
    league = make_league()
    before = compute_features(league).set_index("match_id")

    tampered = league.copy()
    i = tampered.index[tampered["match_id"] == match_id][0]
    tampered.loc[i, ["home_goals", "away_goals", "result"]] = [9, 0, "H"]
    tampered.loc[i, ["home_shots_on_target", "away_shots_on_target"]] = [20, 0]
    after = compute_features(tampered).set_index("match_id")

    match_date = league.loc[i, "match_date"]
    up_to_kickoff = league.loc[league["match_date"] <= match_date, "match_id"]
    pd.testing.assert_frame_equal(before.loc[up_to_kickoff], after.loc[up_to_kickoff])

    # Guard against a vacuous pass: some later Match must see the change.
    later = league.loc[league["match_date"] > match_date, "match_id"]
    assert not before.loc[later].equals(after.loc[later])


def test_first_season_starts_with_no_rolling_history():
    features = compute_features(make_league()).set_index("match_id")
    assert features.loc[0, "home_elo"] == 1500.0
    assert pd.isna(features.loc[0, "home_ppg"])
    assert pd.isna(features.loc[0, "home_goals_for"])


def test_promoted_team_starts_with_relegated_teams_final_form():
    league = make_league()
    first_season = league[league["season"] == "2005-06"]
    features = compute_features(league).set_index("match_id")

    d_matches = first_season[(first_season["home_team"] == "D") | (first_season["away_team"] == "D")].tail(STATS_WINDOW)
    d_goals_for = [m.home_goals if m.home_team == "D" else m.away_goals for m in d_matches.itertuples()]

    second_season = league[league["season"] == "2006-07"]
    e_first = second_season[(second_season["home_team"] == "E") | (second_season["away_team"] == "E")].iloc[0]
    side = "home" if e_first["home_team"] == "E" else "away"
    assert features.loc[e_first["match_id"], f"{side}_goals_for"] == pytest.approx(sum(d_goals_for) / len(d_goals_for))


def test_features_have_no_gaps_once_history_exists():
    features = compute_features(make_league())
    league = make_league()
    second_season_ids = league.loc[league["season"] == "2006-07", "match_id"]
    assert not features.set_index("match_id").loc[second_season_ids, FEATURE_COLUMNS].isna().any().any()
