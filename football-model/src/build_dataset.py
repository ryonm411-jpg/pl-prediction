"""
build_dataset.py
================
PURPOSE:
    Merge parsed match JSONs and team profile data into a single
    flat CSV suitable for scikit-learn training.

HOW IT WORKS (step by step):
    1. Load all match JSONs from  data/raw_matches/
    2. Load team profiles from    data/teams.json
    3. For each match:
       a) Extract the 5 tactical ratings for home & away
       b) Compute DELTA features  (home_rating - away_rating)
       c) Encode formations as simple categorical integers
       d) Determine the match RESULT label (Home=2, Draw=1, Away=0)
    4. Stack everything into a pandas DataFrame
    5. Save to data/datasets/training_data.csv

EXTENSIBILITY HOOKS (for later):
    - Add player-level features (avg squad rating)
    - Add rolling form (last N match ratings)
    - Add style-tag features (one-hot encode high_press, etc.)

BEGINNER NOTES:
    - "Delta" = difference.  If Home pressing = 85 and Away pressing = 60,
      the delta is +25.  Positive means Home is stronger in that area.
    - We map formations to integers because ML models need numbers, not
      strings.  Later you can one-hot encode them for more power.
    - The RESULT column is what the model tries to predict.  It is called
      the "target" or "label" in ML terminology.
"""

import json
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
RAW_MATCHES_DIR = BASE_DIR / "data" / "raw_matches"
TEAMS_FILE = BASE_DIR / "data" / "teams.json"
OUTPUT_DIR = BASE_DIR / "data" / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "training_data.csv"

# The 5 tactical rating categories from your template.
# These MUST match the keys used in your parsed match JSONs.
RATING_CATEGORIES = [
    "Pressing Intensity",
    "Build-up Quality",
    "Chance Creation",
    "Defensive Organization",
    "Defensive Transition Vulnerability",
]

# A lookup table that maps formation strings → integers.
# We start small; add more formations as you encounter them.
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


def main():
    """
    Entry point: orchestrates the full build pipeline.
    """
    print("=" * 60)
    print("BUILD DATASET")
    print("=" * 60)

    # --- Step 1: Load team profiles into a lookup dict ---
    teams_lookup = load_teams(TEAMS_FILE)
    print(f"Loaded {len(teams_lookup)} team profiles.")

    # --- Step 2: Load and process every match JSON ---
    match_files = sorted(RAW_MATCHES_DIR.glob("*.json"))
    if not match_files:
        print("[ERROR] No match JSON files found in data/raw_matches/")
        return

    print(f"Found {len(match_files)} match files.\n")

    rows = []
    skipped = 0

    for mf in match_files:
        row = process_match(mf, teams_lookup)
        if row is not None:
            rows.append(row)
        else:
            skipped += 1

    if not rows:
        print("[ERROR] No valid rows produced. Check your match files.")
        return

    # --- Step 3: Write CSV ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUTPUT_FILE}")
    if skipped:
        print(f"Skipped {skipped} files (missing data).")
    print("Done.")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_match(match_path, teams_lookup):
    """
    Converts one match JSON into a flat dictionary (one CSV row).

    Returns None if the match is missing critical data (ratings or score).
    """
    with open(match_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    ratings = data.get("ratings", {})
    formations = data.get("formations", {})

    # ----- Validate required fields -----
    home_team = meta.get("home_team")
    away_team = meta.get("away_team")
    score_str = meta.get("final_score")

    if not home_team or not away_team:
        print(f"  [SKIP] {match_path.name}: missing team names.")
        return None

    if not score_str:
        print(f"  [SKIP] {match_path.name}: missing final score.")
        return None

    # ----- Parse score → result label -----
    result = parse_result(score_str)
    if result is None:
        print(f"  [SKIP] {match_path.name}: could not parse score '{score_str}'.")
        return None

    # ----- Build feature row -----
    row = {
        "match_id": meta.get("match_id", match_path.stem),
        "date": meta.get("date", ""),
        "home_team": home_team,
        "away_team": away_team,
    }

    # Tactical rating deltas
    home_ratings = ratings.get("home", {})
    away_ratings = ratings.get("away", {})

    for cat in RATING_CATEGORIES:
        h_val = get_rating_value(home_ratings, cat)
        a_val = get_rating_value(away_ratings, cat)
        safe_key = cat.lower().replace(" ", "_").replace("-", "_")

        row[f"home_{safe_key}"] = h_val
        row[f"away_{safe_key}"] = a_val
        row[f"delta_{safe_key}"] = h_val - a_val

    # xG (expected goals) — a strong predictor of true team quality
    home_xg, away_xg = parse_xg(meta.get("xg", ""))
    row["home_xg"] = home_xg
    row["away_xg"] = away_xg
    row["delta_xg"] = home_xg - away_xg

    # Formation encoding
    starting = formations.get("starting", {})
    row["home_formation"] = FORMATION_MAP.get(starting.get("home", ""), 0)
    row["away_formation"] = FORMATION_MAP.get(starting.get("away", ""), 0)

    # Venue (1 = home, 0 = neutral/away from perspective of "home_team")
    venue = meta.get("venue", "").lower()
    row["is_home"] = 1 if venue == "home" else 0

    # Target label
    row["result"] = result

    print(f"  [OK] {match_path.name}: {home_team} vs {away_team} → result={result}")
    return row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_teams(teams_path):
    """
    Load data/teams.json and return a dict keyed by team_name.

    Why a dict?  So we can look up team info in O(1) time later
    instead of scanning a list every time.
    """
    if not teams_path.exists():
        print(f"[WARNING] {teams_path} not found. Team features will be empty.")
        return {}

    with open(teams_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    lookup = {}
    for team in raw.get("teams", []):
        name = team.get("team_name")
        if name:
            lookup[name] = team
    return lookup


def get_rating_value(ratings_dict, category_name, default=50):
    """
    Safely extract a numeric rating from the ratings dict.

    If the category or rating key is missing, returns `default` (50).

    Why 50?  It is the midpoint of our 0-100 scale, meaning
    "average / unknown".  This prevents crashes and avoids
    biasing the model toward high or low values for missing data.
    """
    cat_data = ratings_dict.get(category_name, {})
    val = cat_data.get("rating", default)

    # Ensure it is numeric
    if isinstance(val, (int, float)):
        return val
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_xg(xg_str):
    """
    Parse an xG string like "1.8 – 1.2" into (home_xg, away_xg).

    Handles multiple separator styles:
        "1.8 – 1.2"   (en-dash, from your template)
        "1.8 - 1.2"   (hyphen)
        "1.8 — 1.2"   (em-dash)

    Returns (0.0, 0.0) if the string is empty or unparseable.

    Why xG matters:
        xG measures the QUALITY of chances created, not just the
        final score.  A team that wins 1-0 with xG of 0.3 was lucky;
        a team that loses 0-1 with xG of 2.5 was unlucky.  Over time,
        xG is a better predictor of future results than actual goals.
    """
    if not xg_str or not xg_str.strip():
        return 0.0, 0.0

    try:
        # Normalize separators
        normalized = xg_str.replace("–", "-").replace("—", "-").strip()
        parts = normalized.split("-")
        home_xg = float(parts[0].strip())
        away_xg = float(parts[1].strip())
        return home_xg, away_xg
    except (ValueError, IndexError):
        return 0.0, 0.0


def parse_result(score_str):
    """
    Convert a score string like "2-1" into a result label.

    Labels:
        2 = Home Win
        1 = Draw
        0 = Away Win

    Why these numbers?
        They are arbitrary but conventional.  scikit-learn works
        with integer labels.  We use 2/1/0 so that "higher = better
        for home" which is intuitive when reading feature importances.
    """
    try:
        # Handle various separators: "2-1", "2 - 1", "2–1"
        score_str = score_str.replace("–", "-").replace("—", "-").strip()
        parts = score_str.split("-")
        home_goals = int(parts[0].strip())
        away_goals = int(parts[1].strip())
    except (ValueError, IndexError):
        return None

    if home_goals > away_goals:
        return 2  # Home win
    elif home_goals == away_goals:
        return 1  # Draw
    else:
        return 0  # Away win


if __name__ == "__main__":
    main()
