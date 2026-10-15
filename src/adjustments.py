"""Analyst Adjustments: hand-set Elo shifts for tactical claims (ADR 0002).

They live in adjustments.yaml at the repo root, one entry per claim:

    - id: doku-vs-frimpong
      reason: Doku's 1v1 pace isolates Frimpong, who defends poorly facing the ball
      favours: Man City            # football-data.co.uk team name
      elo: 25                      # Elo points added to `favours`
      season: 2026-27              # expires when this season ends
      against: Liverpool           # optional: only this opponent (omit = any opponent)
      requires_players: [Jérémy Doku, Jeremie Frimpong]   # optional: confirm per Match
      evidence: match_analysis/...md                       # optional
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ADJUSTMENTS_PATH = Path(__file__).resolve().parent.parent / "adjustments.yaml"


@dataclass(frozen=True)
class Adjustment:
    id: str
    reason: str
    favours: str
    elo: float
    season: str
    against: str | None = None
    requires_players: list[str] = field(default_factory=list)
    evidence: str | None = None

    def applies_to(self, season: str, home_team: str, away_team: str) -> bool:
        """Whether this Match is in scope. Required players are confirmed separately."""
        if season != self.season or self.favours not in (home_team, away_team):
            return False
        opponent = away_team if self.favours == home_team else home_team
        return self.against is None or self.against == opponent

    def snapshot(self) -> dict:
        """What gets stored with a Prediction, frozen at the time it is made."""
        return {"id": self.id, "reason": self.reason, "favours": self.favours, "elo": self.elo}


def load(path: Path = ADJUSTMENTS_PATH) -> list[Adjustment]:
    if not path.exists():
        return []
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    adjustments = [Adjustment(**{**e, "season": str(e["season"])}) for e in entries]
    ids = [a.id for a in adjustments]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate adjustment ids in {path.name}: {sorted(duplicates)}")
    return adjustments


def in_scope(adjustments: list[Adjustment], season: str, home_team: str, away_team: str) -> list[Adjustment]:
    return [a for a in adjustments if a.applies_to(season, home_team, away_team)]


def elo_shift(applied: list[Adjustment]) -> dict[str, float]:
    shift: dict[str, float] = {}
    for a in applied:
        shift[a.favours] = shift.get(a.favours, 0.0) + a.elo
    return shift
