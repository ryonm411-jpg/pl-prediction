"""Compute Pre-match Features for every Match and store them in data/epl.db.

Walks through the Matches in date order. For each match date it first records
every team's Elo Rating and recent-form averages, then applies that day's
Results. So a Match's features only ever see Matches from earlier dates
(same-day Matches are treated as simultaneous).

Rolling stats carry over between seasons. A promoted team has no recent EPL
Matches, so its windows start pre-filled with the average of the relegated
teams' final windows: the same "replace like with like" rule Elo uses.

Output: table match_features, one row per Match, joined to matches by match_id.
"""

import sqlite3
import sys
from collections import deque
from pathlib import Path

import pandas as pd

from ratings import EloTable

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "epl.db"

FORM_WINDOW = 5      # Matches used for points per game
STATS_WINDOW = 10    # Matches used for goals and shots on target

# Per-Match stats each team keeps in its rolling window, from its own point of view.
ROLLING_STATS = ["goals_for", "goals_against", "sot_for", "sot_against"]

FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_ppg", "away_ppg",
    "home_goals_for", "home_goals_against", "away_goals_for", "away_goals_against",
    "home_sot_for", "home_sot_against", "away_sot_for", "away_sot_against",
]

POINTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}


class TeamForm:
    """One team's recent Matches: points for form, In-match Stats for the rest."""

    def __init__(self):
        self.points: deque[float] = deque(maxlen=FORM_WINDOW)
        self.stats: deque[dict[str, float]] = deque(maxlen=STATS_WINDOW)

    def snapshot(self) -> dict[str, float]:
        """Averages over the current windows; NaN until the team has any history."""
        out = {"ppg": sum(self.points) / len(self.points) if self.points else float("nan")}
        for stat in ROLLING_STATS:
            values = [m[stat] for m in self.stats]
            out[stat] = sum(values) / len(values) if values else float("nan")
        return out

    def seed(self, average: dict[str, float]) -> None:
        """Fill the windows with copies of an average Match (used for promoted teams)."""
        self.points = deque([average["ppg"]] * FORM_WINDOW, maxlen=FORM_WINDOW)
        record = {stat: average[stat] for stat in ROLLING_STATS}
        self.stats = deque([dict(record) for _ in range(STATS_WINDOW)], maxlen=STATS_WINDOW)


class LeagueState:
    """Every team's Elo Rating and recent form at one point in time.

    Walk Matches through it in date order: `features` reads the state before
    Kickoff, `record` applies a Result afterwards.
    """

    def __init__(self, elo: EloTable | None = None):
        self.elo = elo or EloTable()
        self.form: dict[str, TeamForm] = {}
        self.league: set[str] = set()
        self.season: str | None = None

    def start_season(self, season: str, teams: set[str]) -> None:
        relegated = self.league - teams
        snapshots = [self.form[t].snapshot() for t in relegated]
        for team in teams:
            if team not in self.form or (self.league and team not in self.league):
                self.form[team] = TeamForm()
                if snapshots:
                    self.form[team].seed({k: sum(s[k] for s in snapshots) / len(snapshots) for k in snapshots[0]})
        self.elo.start_season(teams)
        self.league = set(teams)
        self.season = season

    def features(self, home_team: str, away_team: str, elo_shift: dict[str, float] | None = None) -> dict[str, float]:
        """Pre-match Features for a Match between two teams in the current season.

        `elo_shift` adds Elo points per team (Analyst Adjustments) without
        changing the stored ratings.
        """
        shift = elo_shift or {}
        home_elo = self.elo[home_team] + shift.get(home_team, 0.0)
        away_elo = self.elo[away_team] + shift.get(away_team, 0.0)
        home, away = self.form[home_team].snapshot(), self.form[away_team].snapshot()
        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_ppg": home["ppg"],
            "away_ppg": away["ppg"],
            **{f"home_{s}": home[s] for s in ROLLING_STATS},
            **{f"away_{s}": away[s] for s in ROLLING_STATS},
        }

    def record(self, m) -> None:
        """Apply one played Match (any object with the matches-table fields)."""
        self.elo.update(m.home_team, m.away_team, m.result)
        home_points, away_points = POINTS[m.result]
        self.form[m.home_team].points.append(home_points)
        self.form[m.away_team].points.append(away_points)
        self.form[m.home_team].stats.append({
            "goals_for": m.home_goals, "goals_against": m.away_goals,
            "sot_for": m.home_shots_on_target, "sot_against": m.away_shots_on_target,
        })
        self.form[m.away_team].stats.append({
            "goals_for": m.away_goals, "goals_against": m.home_goals,
            "sot_for": m.away_shots_on_target, "sot_against": m.home_shots_on_target,
        })


def walk(matches: pd.DataFrame, state: LeagueState, on_day=None) -> LeagueState:
    """Feed Matches through `state` season by season and date by date.

    `on_day(state, day)` runs before each date's Results are applied, which is
    where Pre-match Features are read.
    """
    matches = matches.sort_values(["match_date", "match_id"])
    for season, season_matches in matches.groupby("season", sort=True):
        state.start_season(season, set(season_matches["home_team"]) | set(season_matches["away_team"]))
        for _, day in season_matches.groupby("match_date", sort=True):
            if on_day:
                on_day(state, day)
            for m in day.itertuples():
                state.record(m)
    return state


def compute_features(matches: pd.DataFrame, elo: EloTable | None = None) -> pd.DataFrame:
    """Return one row of Pre-match Features per Match.

    `matches` needs: match_id, season, match_date, home_team, away_team,
    home_goals, away_goals, result, home_shots_on_target, away_shots_on_target.
    Pass a fresh `elo` table to try Elo settings other than the defaults.
    """
    rows = []

    def read_day(state: LeagueState, day: pd.DataFrame) -> None:
        for m in day.itertuples():
            rows.append({"match_id": m.match_id, **state.features(m.home_team, m.away_team)})

    walk(matches, LeagueState(elo), on_day=read_day)
    return pd.DataFrame(rows, columns=["match_id", *FEATURE_COLUMNS])


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        matches = pd.read_sql("SELECT * FROM matches", conn)
        features = compute_features(matches)
        features.to_sql("match_features", conn, if_exists="replace", index=False)
    print(f"Wrote Pre-match Features for {len(features)} matches to {DB_PATH}")


if __name__ == "__main__":
    sys.exit(main())
