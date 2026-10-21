"""Predict an upcoming Match, apply Analyst Adjustments, and log the Prediction.

    predict.py match "Man City" "Liverpool" --odds 2.10 3.60 3.40
    predict.py fixtures
    predict.py report

`match` rebuilds the league's current state from data/epl.db (run fetch_data.py
and build_db.py first for the latest Results). It shows the model's probabilities,
asks which in-scope Analyst Adjustments apply, shows the adjusted probabilities
and the Team Profile notes, and logs the Prediction before Kickoff.

`fixtures` does the same for every upcoming EPL fixture football-data.co.uk
lists, using its average odds as the market.

`report` scores every logged Prediction whose Result is now in the database.
"""

import argparse
import difflib
import io
import sqlite3
import sys
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import adjustments as adj
import predictions
from build_dataset import DB_PATH, FEATURE_COLUMNS, LeagueState, walk
from train_model import LIVE_MODEL_PATH

TEAMS_DIR = Path(__file__).resolve().parent.parent / "teams"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"  # upcoming matches, all leagues, with odds

# football-data.co.uk name -> Team Profile file name, where they differ.
PROFILE_NAMES = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton & Hove Albion",
    "Leeds": "Leeds United",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}
PROFILE_SECTIONS = ["## Team Strengths & Weaknesses", "## Known Matchup Vulnerabilities"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Team Profiles contain names like Ødegaard
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        choice = p.add_mutually_exclusive_group()
        choice.add_argument("--apply", default=None, help="comma-separated adjustment ids to apply without asking")
        choice.add_argument("--no-adjust", action="store_true", help="apply no Analyst Adjustments")
        p.add_argument("--dry-run", action="store_true", help="show Predictions without logging them")

    m = sub.add_parser("match", help="predict and log one upcoming Match")
    m.add_argument("home")
    m.add_argument("away")
    m.add_argument("--odds", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"),
                   help="decimal odds, to compare against the market")
    add_common(m)
    f = sub.add_parser("fixtures", help="predict and log every upcoming EPL fixture on football-data.co.uk")
    f.add_argument("--profiles", action="store_true", help="also print Team Profile notes")
    add_common(f)
    sub.add_parser("report", help="score logged Predictions against Results")
    args = parser.parse_args()

    if args.command == "match":
        ctx = load_context()
        home, away = resolve_team(args.home, ctx.state.league), resolve_team(args.away, ctx.state.league)
        if not forecast(ctx, home, away, args.odds, args, show_profiles=True):
            sys.exit(1)
    elif args.command == "fixtures":
        predict_fixtures(args)
    else:
        print_report()


class Context:
    """Everything a forecast needs: played Matches, the league's current state, the live model."""

    def __init__(self, matches: pd.DataFrame, state: LeagueState, model):
        self.matches, self.state, self.model = matches, state, model


def load_context() -> Context:
    with sqlite3.connect(DB_PATH) as conn:
        matches = pd.read_sql("SELECT * FROM matches", conn)
    if not LIVE_MODEL_PATH.exists():
        sys.exit("No live model yet: run train_model.py, evaluate.py, then train_model.py --live.")
    return Context(matches, walk(matches, LeagueState()), joblib.load(LIVE_MODEL_PATH))


def forecast(ctx: Context, home: str, away: str, odds, args, show_profiles: bool) -> bool:
    """Show and (unless --dry-run) log one Prediction. `odds` are decimal (home, draw, away).

    Returns False when the Match has already been played.
    """
    season = ctx.state.season
    m = ctx.matches
    played = m[(m["season"] == season) & (m["home_team"] == home) & (m["away_team"] == away)]
    if len(played):
        print(f"{home} vs {away} was already played on {played.iloc[0]['match_date']}; "
              "Predictions must be made before Kickoff.")
        return False

    model_p = predict_one(ctx.model, ctx.state.features(home, away))
    applied = choose_adjustments(adj.in_scope(adj.load(), season, home, away), args)
    adjusted_p = predict_one(ctx.model, ctx.state.features(home, away, adj.elo_shift(applied)))

    market_p = None
    if odds is not None and all(pd.notna(o) for o in odds):
        implied = 1.0 / np.array([odds[2], odds[1], odds[0]])  # away, draw, home
        market_p = implied / implied.sum()

    print(f"\n{home} vs {away} ({season}, data up to {m['match_date'].max()})\n")
    rows = [("Model", model_p)]
    if applied:
        rows.append(("Model + Analyst Adjustments", adjusted_p))
    if market_p is not None:
        rows.append(("Market", market_p))
    print(f"{'':30} {'Home':>6} {'Draw':>6} {'Away':>6}")
    for name, p in rows:
        print(f"{name:30} {p[2]:6.1%} {p[1]:6.1%} {p[0]:6.1%}")

    if show_profiles:
        for team in (home, away):
            print_profile(team)

    if args.dry_run:
        print("Dry run: Prediction not logged.")
        return True
    with predictions.connect() as conn:
        predictions.log(conn, season, home, away, model_p, adjusted_p, market_p, [a.snapshot() for a in applied])
    print(f"Logged to {predictions.PREDICTIONS_DB}")
    return True


def upcoming_epl_fixtures(url: str | None = None) -> pd.DataFrame:
    with urllib.request.urlopen(url or FIXTURES_URL, timeout=30) as resp:
        fixtures = pd.read_csv(io.BytesIO(resp.read()), encoding="utf-8-sig")
    return fixtures[fixtures["Div"] == "E0"]


def predict_fixtures(args) -> None:
    fixtures = upcoming_epl_fixtures()
    if fixtures.empty:
        print("football-data.co.uk lists no upcoming EPL fixtures yet; they usually appear a few days "
              "before each matchweek.")
        return
    ctx = load_context()
    for f in fixtures.itertuples():
        home, away = resolve_team(f.HomeTeam, ctx.state.league), resolve_team(f.AwayTeam, ctx.state.league)
        print(f"\n--- {f.Date} {getattr(f, 'Time', '')} ---")
        forecast(ctx, home, away, (f.AvgH, f.AvgD, f.AvgA), args, show_profiles=args.profiles)


def resolve_team(name: str, league: set[str]) -> str:
    if name in league:
        return name
    close = difflib.get_close_matches(name, sorted(league), n=3, cutoff=0.4)
    sys.exit(f"Unknown team '{name}' for the current season." + (f" Did you mean: {', '.join(close)}?" if close else
             f" Teams: {', '.join(sorted(league))}"))


def predict_one(model, features: dict[str, float]) -> np.ndarray:
    return model.predict_proba(pd.DataFrame([features])[FEATURE_COLUMNS])[0]


def choose_adjustments(candidates: list[adj.Adjustment], args) -> list[adj.Adjustment]:
    if args.no_adjust or not candidates:
        return []
    if args.apply is not None:
        wanted = {i.strip() for i in args.apply.split(",") if i.strip()}
        unknown = wanted - {a.id for a in candidates}
        if unknown:
            sys.exit(f"Not in scope for this Match: {', '.join(sorted(unknown))}. "
                     f"In scope: {', '.join(a.id for a in candidates)}")
        return [a for a in candidates if a.id in wanted]

    chosen = []
    print("\nAnalyst Adjustments in scope:")
    for a in candidates:
        print(f"\n  [{a.id}] {a.elo:+g} Elo to {a.favours}: {a.reason}")
        question = "  Apply?"
        if a.requires_players:
            everyone = "both" if len(a.requires_players) == 2 else "all"
            question = f"  Are {' and '.join(a.requires_players)} {everyone} playing?"
        if input(f"{question} [y/N] ").strip().lower() == "y":
            chosen.append(a)
    return chosen


def print_profile(team: str) -> None:
    path = TEAMS_DIR / f"{PROFILE_NAMES.get(team, team)}.md"
    print(f"\n=== {team}: Team Profile ===")
    if not path.exists():
        print("(no Team Profile)")
        return
    text = path.read_text(encoding="utf-8")
    for heading in PROFILE_SECTIONS:
        start = text.find(heading)
        if start == -1:
            continue
        end = min(i for i in (text.find("\n---", start), text.find("\n## ", start + 1), len(text)) if i != -1)
        print(text[start:end].strip() + "\n")


def print_report() -> None:
    with predictions.connect() as conn:
        df = predictions.scored(conn, DB_PATH)
        pending = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] - len(df)
    if df.empty:
        print(f"No logged Predictions have a Result yet ({pending} waiting).")
        return
    overall, per_adjustment = predictions.report(df)
    print(f"Scored Predictions: {len(df)} ({pending} still waiting for a Result)\n")
    print(overall.to_string(index=False, float_format="%.4f"))
    if not per_adjustment.empty:
        print("\nPer Analyst Adjustment (log_loss_change < 0 means it helped):")
        print(per_adjustment.to_string(index=False, float_format="%.4f"))
    if len(df) < 100:
        print(f"\nOnly {len(df)} Predictions: read these differences as anecdotes, not evidence.")


if __name__ == "__main__":
    sys.exit(main())
