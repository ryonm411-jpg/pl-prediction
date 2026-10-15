"""Predict an upcoming Match, apply Analyst Adjustments, and log the Prediction.

    predict.py match "Man City" "Liverpool" --odds 2.10 3.60 3.40
    predict.py report

`match` rebuilds the league's current state from data/epl.db (run fetch_data.py
and build_db.py first for the latest Results). It shows the model's probabilities,
asks which in-scope Analyst Adjustments apply, shows the adjusted probabilities
and the Team Profile notes, and logs the Prediction before Kickoff.

`report` scores every logged Prediction whose Result is now in the database.
"""

import argparse
import difflib
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import adjustments as adj
import predictions
from build_dataset import DB_PATH, FEATURE_COLUMNS, LeagueState, walk
from train_model import LIVE_MODEL_PATH

TEAMS_DIR = Path(__file__).resolve().parent.parent / "teams"

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
    m = sub.add_parser("match", help="predict and log one upcoming Match")
    m.add_argument("home")
    m.add_argument("away")
    m.add_argument("--odds", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"),
                   help="decimal odds, to compare against the market")
    choice = m.add_mutually_exclusive_group()
    choice.add_argument("--apply", default=None, help="comma-separated adjustment ids to apply without asking")
    choice.add_argument("--no-adjust", action="store_true", help="apply no Analyst Adjustments")
    m.add_argument("--dry-run", action="store_true", help="show the Prediction without logging it")
    sub.add_parser("report", help="score logged Predictions against Results")
    args = parser.parse_args()

    if args.command == "match":
        predict_match(args)
    else:
        print_report()


def predict_match(args) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        matches = pd.read_sql("SELECT * FROM matches", conn)
    state = walk(matches, LeagueState())
    home, away = resolve_team(args.home, state.league), resolve_team(args.away, state.league)
    season = state.season

    played = matches[(matches["season"] == season) & (matches["home_team"] == home) & (matches["away_team"] == away)]
    if len(played):
        sys.exit(f"{home} vs {away} was already played on {played.iloc[0]['match_date']}; "
                 "Predictions must be made before Kickoff.")

    if not LIVE_MODEL_PATH.exists():
        sys.exit("No live model yet: run train_model.py, evaluate.py, then train_model.py --live.")
    model = joblib.load(LIVE_MODEL_PATH)
    model_p = predict_one(model, state.features(home, away))

    applied = choose_adjustments(adj.in_scope(adj.load(), season, home, away), args)
    adjusted_p = predict_one(model, state.features(home, away, adj.elo_shift(applied)))

    market_p = None
    if args.odds:
        implied = 1.0 / np.array([args.odds[2], args.odds[1], args.odds[0]])  # away, draw, home
        market_p = implied / implied.sum()

    print(f"\n{home} vs {away} ({season}, data up to {matches['match_date'].max()})\n")
    rows = [("Model", model_p)]
    if applied:
        rows.append(("Model + Analyst Adjustments", adjusted_p))
    if market_p is not None:
        rows.append(("Market (your odds)", market_p))
    print(f"{'':30} {'Home':>6} {'Draw':>6} {'Away':>6}")
    for name, p in rows:
        print(f"{name:30} {p[2]:6.1%} {p[1]:6.1%} {p[0]:6.1%}")

    for team in (home, away):
        print_profile(team)

    if args.dry_run:
        print("\nDry run: Prediction not logged.")
        return
    with predictions.connect() as conn:
        predictions.log(conn, season, home, away, model_p, adjusted_p, market_p, [a.snapshot() for a in applied])
    print(f"\nLogged to {predictions.PREDICTIONS_DB}")


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
