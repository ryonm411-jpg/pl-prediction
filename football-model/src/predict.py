"""
predict.py
==========
PURPOSE:
    Predict the outcome of a future match by providing the tactical
    ratings for both teams.

HOW TO USE:
    Option A — Command line:
        python src/predict.py

    Option B — From another script:
        from predict import predict_match
        result = predict_match(home_ratings, away_ratings, ...)

TWO MODES:
    1. INTERACTIVE: The script prompts you for team names and uses
       their data from the latest parsed match JSON.
    2. MANUAL: You supply rating values directly (useful for
       "what-if" scenarios like "what if we press harder?").

BEGINNER NOTES:
    - The model outputs probabilities for 3 outcomes:
      Home Win, Draw, Away Win.  These SHOULD sum to ~100%.
    - Higher confidence = one probability is much larger than others.
    - The model is only as good as your ratings.  Garbage in = garbage out.

EXTENSIBILITY:
    - Accept a match Markdown file as input (parse → predict).
    - Add betting value calculations (compare to bookmaker odds).
    - Add historical comparison ("last 5 times these teams played...").
"""

import json
import sys
import numpy as np
from pathlib import Path
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
MODEL_FILE = BASE_DIR / "data" / "models" / "football_model.pkl"
FEATURES_FILE = BASE_DIR / "data" / "models" / "feature_columns.json"
TEAMS_FILE = BASE_DIR / "data" / "teams.json"
RAW_MATCHES_DIR = BASE_DIR / "data" / "raw_matches"

# Must match build_dataset.py exactly
RATING_CATEGORIES = [
    "Pressing Intensity",
    "Build-up Quality",
    "Chance Creation",
    "Defensive Organization",
    "Defensive Transition Vulnerability",
]

FORMATION_MAP = {
    "4-3-3":   1,
    "4-2-3-1": 2,
    "4-4-2":   3,
    "3-5-2":   4,
    "3-4-3":   5,
    "4-1-4-1": 6,
    "4-2-4":   7,
    "2-3-5":   8,
    "5-3-2":   9,
    "4-3-1-2": 10,
    "3-4-2-1": 11,
    "4-1-2-1-2": 12,
}

LABEL_NAMES = {0: "Away Win", 1: "Draw", 2: "Home Win"}


def main():
    """
    Interactive prediction mode.
    """
    print("=" * 60)
    print("MATCH PREDICTION")
    print("=" * 60)

    # Load model
    if not MODEL_FILE.exists():
        print(f"[ERROR] Model not found: {MODEL_FILE}")
        print("Run train_model.py first.")
        return

    model = joblib.load(MODEL_FILE)

    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    print(f"Model loaded ({len(feature_cols)} features).\n")

    # --- Choose input mode ---
    print("Choose input mode:")
    print("  1) Enter team names (uses latest match data)")
    print("  2) Enter ratings manually")
    choice = input("\nChoice [1/2]: ").strip()

    if choice == "2":
        predict_manual(model, feature_cols)
    else:
        predict_from_data(model, feature_cols)


def predict_from_data(model, feature_cols):
    """
    Look up the most recent match data for each team and use those
    ratings to predict a hypothetical matchup.
    """
    # Build a lookup of latest ratings per team from raw_matches
    team_ratings = build_latest_ratings_lookup()

    if not team_ratings:
        print("[ERROR] No match data found. Parse some matches first.")
        return

    available = sorted(team_ratings.keys())
    print(f"\nAvailable teams ({len(available)}):")
    for t in available:
        print(f"  - {t}")

    home_name = input("\nHome team: ").strip()
    away_name = input("Away team: ").strip()

    if home_name not in team_ratings:
        print(f"[ERROR] '{home_name}' not found in parsed match data.")
        return
    if away_name not in team_ratings:
        print(f"[ERROR] '{away_name}' not found in parsed match data.")
        return

    home_r = team_ratings[home_name]
    away_r = team_ratings[away_name]

    home_formation = input(f"Home formation (default {home_r.get('formation', '4-3-3')}): ").strip()
    away_formation = input(f"Away formation (default {away_r.get('formation', '4-2-3-1')}): ").strip()

    if not home_formation:
        home_formation = home_r.get("formation", "4-3-3")
    if not away_formation:
        away_formation = away_r.get("formation", "4-2-3-1")

    # xG — use latest match xG as a baseline, or let user override
    default_home_xg = home_r.get("xg", 1.2)
    default_away_xg = away_r.get("xg", 1.2)
    home_xg_input = input(f"Home expected xG (default {default_home_xg}): ").strip()
    away_xg_input = input(f"Away expected xG (default {default_away_xg}): ").strip()
    home_xg = float(home_xg_input) if home_xg_input else default_home_xg
    away_xg = float(away_xg_input) if away_xg_input else default_away_xg

    feature_vector = build_feature_vector(
        home_r["ratings"], away_r["ratings"],
        home_formation, away_formation,
        is_home=1,
        feature_cols=feature_cols,
        home_xg=home_xg, away_xg=away_xg
    )

    run_prediction(model, feature_vector, home_name, away_name, feature_cols)


def predict_manual(model, feature_cols):
    """
    Prompt the user for each rating value manually.
    """
    print("\n--- Enter Home Team Ratings (0-100) ---")
    home_ratings = {}
    for cat in RATING_CATEGORIES:
        val = input(f"  {cat}: ").strip()
        home_ratings[cat] = int(val) if val else 50

    print("\n--- Enter Away Team Ratings (0-100) ---")
    away_ratings = {}
    for cat in RATING_CATEGORIES:
        val = input(f"  {cat}: ").strip()
        away_ratings[cat] = int(val) if val else 50

    home_formation = input("\nHome formation (e.g., 4-3-3): ").strip() or "4-3-3"
    away_formation = input("Away formation (e.g., 4-2-3-1): ").strip() or "4-2-3-1"

    # xG input
    home_xg_input = input("\nHome expected xG (e.g., 1.5, or press Enter for 1.2): ").strip()
    away_xg_input = input("Away expected xG (e.g., 1.0, or press Enter for 1.2): ").strip()
    home_xg = float(home_xg_input) if home_xg_input else 1.2
    away_xg = float(away_xg_input) if away_xg_input else 1.2

    feature_vector = build_feature_vector(
        home_ratings, away_ratings,
        home_formation, away_formation,
        is_home=1,
        feature_cols=feature_cols,
        home_xg=home_xg, away_xg=away_xg
    )

    run_prediction(model, feature_vector, "Home Team", "Away Team", feature_cols)


def run_prediction(model, feature_vector, home_name, away_name, feature_cols):
    """
    Execute prediction and display results.
    """
    X = np.array([feature_vector])
    proba = model.predict_proba(X)[0]
    pred_class = model.predict(X)[0]

    classes = model.classes_
    prob_dict = {int(classes[i]): proba[i] for i in range(len(classes))}

    home_pct = prob_dict.get(2, 0) * 100
    draw_pct = prob_dict.get(1, 0) * 100
    away_pct = prob_dict.get(0, 0) * 100

    print("\n" + "=" * 50)
    print(f"  {home_name}  vs  {away_name}")
    print("=" * 50)
    print(f"  Home Win:  {home_pct:>5.1f}%")
    print(f"  Draw:      {draw_pct:>5.1f}%")
    print(f"  Away Win:  {away_pct:>5.1f}%")
    print("-" * 50)
    print(f"  Prediction: {LABEL_NAMES.get(pred_class, '?')}")
    print("=" * 50)

    # Confidence assessment
    max_prob = max(home_pct, draw_pct, away_pct)
    if max_prob > 70:
        confidence = "HIGH"
    elif max_prob > 50:
        confidence = "MEDIUM"
    else:
        confidence = "LOW (close match)"
    print(f"  Confidence: {confidence}")


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_feature_vector(home_ratings, away_ratings, home_formation, away_formation,
                         is_home, feature_cols, home_xg=1.2, away_xg=1.2):
    """
    Construct a feature vector that matches the training data columns exactly.

    This is the CRITICAL function — the feature order MUST match what
    build_dataset.py produced, otherwise the model will get nonsense input.
    """
    row = {}

    for cat in RATING_CATEGORIES:
        safe_key = cat.lower().replace(" ", "_").replace("-", "_")

        h_val = home_ratings.get(cat, 50)
        a_val = away_ratings.get(cat, 50)

        # Ensure numeric
        if isinstance(h_val, dict):
            h_val = h_val.get("rating", 50)
        if isinstance(a_val, dict):
            a_val = a_val.get("rating", 50)

        h_val = float(h_val) if h_val is not None else 50.0
        a_val = float(a_val) if a_val is not None else 50.0

        row[f"home_{safe_key}"] = h_val
        row[f"away_{safe_key}"] = a_val
        row[f"delta_{safe_key}"] = h_val - a_val

    # xG features
    row["home_xg"] = float(home_xg)
    row["away_xg"] = float(away_xg)
    row["delta_xg"] = float(home_xg) - float(away_xg)

    row["home_formation"] = FORMATION_MAP.get(home_formation, 0)
    row["away_formation"] = FORMATION_MAP.get(away_formation, 0)
    row["is_home"] = is_home

    # Build vector in the EXACT order the model expects
    vector = []
    for col in feature_cols:
        vector.append(row.get(col, 0))

    return vector


def build_latest_ratings_lookup():
    """
    Scan all parsed match JSONs and extract the LATEST ratings
    for each team (whether they played home or away).

    Returns dict like:
        {"Blue FC": {"ratings": {...}, "formation": "4-3-3"}, ...}
    """
    if not RAW_MATCHES_DIR.exists():
        return {}

    team_data = {}

    for mf in sorted(RAW_MATCHES_DIR.glob("*.json")):
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("meta", {})
        ratings = data.get("ratings", {})
        formations = data.get("formations", {})

        home_team = meta.get("home_team")
        away_team = meta.get("away_team")

        starting = formations.get("starting", {})

        # Parse xG from the match (e.g. "1.8 – 1.2")
        home_xg, away_xg = _parse_xg_string(meta.get("xg", ""))

        if home_team and ratings.get("home"):
            team_data[home_team] = {
                "ratings": ratings["home"],
                "formation": starting.get("home", "4-3-3"),
                "xg": home_xg,
            }

        if away_team and ratings.get("away"):
            team_data[away_team] = {
                "ratings": ratings["away"],
                "formation": starting.get("away", "4-2-3-1"),
                "xg": away_xg,
            }

    return team_data


def _parse_xg_string(xg_str):
    """Parse xG string like '1.8 – 1.2' → (1.8, 1.2). Returns (1.2, 1.2) on failure."""
    if not xg_str or not xg_str.strip():
        return 1.2, 1.2
    try:
        normalized = xg_str.replace("\u2013", "-").replace("\u2014", "-").strip()
        parts = normalized.split("-")
        return float(parts[0].strip()), float(parts[1].strip())
    except (ValueError, IndexError):
        return 1.2, 1.2


# ---------------------------------------------------------------------------
# Non-interactive API (for use by other scripts)
# ---------------------------------------------------------------------------

def predict_match(home_ratings, away_ratings, home_formation="4-3-3",
                  away_formation="4-2-3-1", is_home=1, home_xg=1.2, away_xg=1.2):
    """
    Programmatic prediction interface.

    Args:
        home_ratings: dict of {category_name: rating_value}
        away_ratings: dict of {category_name: rating_value}
        home_formation: string like "4-3-3"
        away_formation: string like "4-2-3-1"
        is_home: 1 if home, 0 if neutral
        home_xg: expected goals for home team
        away_xg: expected goals for away team

    Returns:
        dict with keys: home_win, draw, away_win, prediction
    """
    model = joblib.load(MODEL_FILE)
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    feature_vector = build_feature_vector(
        home_ratings, away_ratings,
        home_formation, away_formation,
        is_home, feature_cols,
        home_xg=home_xg, away_xg=away_xg
    )

    X = np.array([feature_vector])
    proba = model.predict_proba(X)[0]
    pred_class = model.predict(X)[0]

    classes = model.classes_
    prob_dict = {int(classes[i]): proba[i] for i in range(len(classes))}

    return {
        "home_win": prob_dict.get(2, 0),
        "draw": prob_dict.get(1, 0),
        "away_win": prob_dict.get(0, 0),
        "prediction": LABEL_NAMES.get(pred_class, "Unknown"),
    }


if __name__ == "__main__":
    main()
