"""
train_model.py
==============
PURPOSE:
    Train a probabilistic classifier on the tactical features
    produced by build_dataset.py.

MODEL CHOICE — RandomForestClassifier:
    - Handles non-linear relationships (e.g., pressing delta matters
      more when combined with high chance creation).
    - Provides built-in feature importance (so you can see WHICH
      tactical ratings drive predictions the most).
    - Robust to small datasets — doesn't overfit as easily as deep
      learning.
    - Easy to swap later for XGBoost / LightGBM without changing
      the rest of the pipeline.

CALIBRATION:
    Raw RandomForest probabilities are often poorly calibrated
    (e.g., it says "70% home win" but the true rate is only 55%).
    We wrap the model in CalibratedClassifierCV to fix this.

OUTPUT:
    data/models/football_model.pkl   — the trained (calibrated) model
    data/models/feature_columns.json — list of feature column names
                                       (needed by predict.py)

BEGINNER NOTES:
    - "Fitting" = training.  model.fit(X, y) means "learn patterns
      from features X that predict labels y".
    - "pkl" = pickle, Python's way of saving objects to disk.
    - We use joblib instead of pickle because it handles numpy
      arrays more efficiently.
    - max_depth=8 limits tree depth to prevent overfitting on small
      datasets.  Increase this as you add more match data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATASET_FILE = BASE_DIR / "data" / "datasets" / "training_data.csv"
MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_FILE = MODEL_DIR / "football_model.pkl"
FEATURES_FILE = MODEL_DIR / "feature_columns.json"

# Columns that are NOT features (identifiers + the target)
NON_FEATURE_COLS = ["match_id", "date", "home_team", "away_team", "result"]

# Model hyperparameters — tuned conservatively for small datasets.
# Increase n_estimators and max_depth as your dataset grows.
RF_PARAMS = {
    "n_estimators": 200,       # number of trees in the forest
    "max_depth": 8,            # max splits per tree (prevents overfitting)
    "min_samples_split": 3,    # minimum samples to split a node
    "min_samples_leaf": 2,     # minimum samples in a leaf
    "class_weight": "balanced", # handles imbalanced classes (fewer draws)
    "random_state": 42,        # reproducibility
    "n_jobs": -1,              # use all CPU cores
}

TEST_SIZE = 0.2  # 20% of data held out for evaluation


def main():
    print("=" * 60)
    print("TRAIN MODEL")
    print("=" * 60)

    # --- Step 1: Load dataset ---
    if not DATASET_FILE.exists():
        print(f"[ERROR] Dataset not found: {DATASET_FILE}")
        print("Run build_dataset.py first.")
        return

    df = pd.read_csv(DATASET_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    # --- Step 2: Separate features and target ---
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df["result"].values

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # --- Step 3: Train/test split ---
    # With very small datasets (< 20 rows), we skip the split and train
    # on everything.  The model won't be properly validated, but it will
    # at least produce predictions.  As your dataset grows, the split
    # becomes meaningful.
    if len(df) < 10:
        print(f"\n[NOTE] Only {len(df)} samples — too few for a meaningful split.")
        print("       Training on ALL data.  Add more match analyses for real validation.\n")
        X_train, y_train = X, y
        X_test, y_test = X, y  # evaluate on training data (just for sanity)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, stratify=y
        )
        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # --- Step 4: Train the model ---
    print("\nTraining RandomForest...")
    base_model = RandomForestClassifier(**RF_PARAMS)
    base_model.fit(X_train, y_train)

    train_acc = base_model.score(X_train, y_train)
    print(f"Training accuracy: {train_acc:.3f}")

    # --- Step 5: Calibrate probabilities ---
    # CalibratedClassifierCV re-fits on the training data using
    # cross-validation to produce better-calibrated probabilities.
    # With very small datasets, we use fewer cv folds.
    n_cv = min(3, len(np.unique(y_train)))
    if len(X_train) >= 6:
        print("Calibrating probabilities...")
        calibrated_model = CalibratedClassifierCV(
            base_model, cv=n_cv, method="sigmoid"
        )
        calibrated_model.fit(X_train, y_train)
        final_model = calibrated_model
    else:
        print("[NOTE] Too few samples for calibration; using raw model.")
        final_model = base_model

    # --- Step 6: Save artifacts ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    with open(FEATURES_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Feature columns saved to {FEATURES_FILE}")

    # --- Step 7: Quick feature importance (from base model) ---
    print("\n--- Feature Importance (Top 10) ---")
    importances = base_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx[:10]:
        print(f"  {feature_cols[i]:45s} {importances[i]:.4f}")

    print("\nDone.  Run evaluate.py for detailed metrics.")


if __name__ == "__main__":
    main()
