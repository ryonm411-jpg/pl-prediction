import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import adjustments as adj  # noqa: E402
from build_dataset import LeagueState  # noqa: E402

PAIRING = adj.Adjustment(id="doku", reason="pace vs slow full-back", favours="Man City", elo=25,
                         season="2026-27", against="Liverpool", requires_players=["Doku", "Frimpong"])
ANY_OPPONENT = adj.Adjustment(id="lfc-right-side", reason="right side exposed in transition",
                              favours="Brighton", elo=-10, season="2026-27")


@pytest.mark.parametrize("home, away", [("Man City", "Liverpool"), ("Liverpool", "Man City")])
def test_pairing_applies_home_or_away(home, away):
    assert PAIRING.applies_to("2026-27", home, away)


def test_pairing_ignores_other_opponents():
    assert not PAIRING.applies_to("2026-27", "Man City", "Brighton")


def test_any_opponent_claim_applies_to_every_match_of_that_team():
    assert ANY_OPPONENT.applies_to("2026-27", "Brighton", "Everton")
    assert ANY_OPPONENT.applies_to("2026-27", "Arsenal", "Brighton")
    assert not ANY_OPPONENT.applies_to("2026-27", "Arsenal", "Everton")


def test_adjustment_expires_after_its_season():
    assert not PAIRING.applies_to("2027-28", "Man City", "Liverpool")


def test_elo_shift_sums_per_team():
    extra = adj.Adjustment(id="x", reason="r", favours="Man City", elo=5, season="2026-27")
    assert adj.elo_shift([PAIRING, extra, ANY_OPPONENT]) == {"Man City": 30, "Brighton": -10}


def test_load_reads_yaml_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "adjustments.yaml"
    entry = "- {id: a, reason: r, favours: Arsenal, elo: 10, season: 2026-27}\n"
    path.write_text(entry, encoding="utf-8")
    loaded = adj.load(path)
    assert loaded[0].season == "2026-27" and loaded[0].against is None

    path.write_text(entry * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        adj.load(path)


def test_elo_shift_moves_only_elo_features_and_leaves_ratings_untouched():
    state = LeagueState()
    state.start_season("2026-27", {"Man City", "Liverpool"})
    before = state.features("Man City", "Liverpool")
    shifted = state.features("Man City", "Liverpool", {"Man City": 25})

    assert shifted["home_elo"] == before["home_elo"] + 25
    assert shifted["elo_diff"] == before["elo_diff"] + 25
    assert shifted["away_elo"] == before["away_elo"]
    assert state.elo["Man City"] == before["home_elo"]  # stored rating unchanged
