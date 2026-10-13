"""Elo Ratings for EPL teams.

Each Match moves points from the loser to the winner (half a win for a draw),
scaled by how surprising the Result was. The home side gets a fixed Elo bonus
when computing the expected score, because home teams win far more often than
away teams of equal strength.

Between seasons:
- Promoted teams start at the average Elo of the teams they replaced, so the
  league average stays at INITIAL_ELO.
- Every team is pulled SEASON_REGRESSION of the way back to INITIAL_ELO, since
  squads and managers change over the summer.
"""

INITIAL_ELO = 1500.0
# Chosen with src/tune_elo.py on the validation seasons. K of 30-40 beat 10-25;
# home advantage moved validation log loss by under 0.0005 across 30-90 (the
# model learns its own home effect), so it stays at a conventional 60.
K_FACTOR = 30.0
HOME_ADVANTAGE = 60.0
SEASON_REGRESSION = 0.2

RESULT_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}


def expected_home_score(home_elo: float, away_elo: float, home_advantage: float = HOME_ADVANTAGE) -> float:
    """Home team's expected score (win = 1, draw = 0.5) before the Match."""
    return 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400.0))


class EloTable:
    def __init__(self, k_factor: float = K_FACTOR, home_advantage: float = HOME_ADVANTAGE,
                 season_regression: float = SEASON_REGRESSION):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.season_regression = season_regression
        self.ratings: dict[str, float] = {}
        self.league: set[str] = set()

    def __getitem__(self, team: str) -> float:
        return self.ratings[team]

    def start_season(self, teams: set[str]) -> None:
        """Set up the ratings for a new season's 20 teams."""
        if self.league:
            relegated = self.league - teams
            promoted = teams - self.league
            inherited = sum(self.ratings[t] for t in relegated) / len(relegated) if relegated else INITIAL_ELO
            for team in promoted:
                self.ratings[team] = inherited
            for team in teams:
                self.ratings[team] += self.season_regression * (INITIAL_ELO - self.ratings[team])
        else:
            for team in teams:
                self.ratings[team] = INITIAL_ELO
        self.league = set(teams)

    def update(self, home: str, away: str, result: str) -> None:
        """Apply one Result. Points gained by one team are lost by the other."""
        expected = expected_home_score(self.ratings[home], self.ratings[away], self.home_advantage)
        delta = self.k_factor * (RESULT_SCORE[result] - expected)
        self.ratings[home] += delta
        self.ratings[away] -= delta
