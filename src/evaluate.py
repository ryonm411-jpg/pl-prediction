"""
evaluate.py
===========
PURPOSE:
    Evaluate the trained model's performance on the dataset.

WHAT IT PRODUCES:
    1. Classification Report — precision, recall, F1-score per class
    2. Confusion Matrix       — where predictions go right/wrong
    3. Feature Importance Bar Chart (saved as PNG)
    4. Probability Distribution — how confident the model is

BEGINNER NOTES:
    - "Precision" = of all the times the model predicted "Home Win",
      how often was it actually a Home Win?
    - "Recall" = of all actual Home Wins, how many did the model catch?
    - "F1" = the harmonic mean of precision and recall.  Good single
      metric when you care about both.
    - With very few matches (< 20), these metrics won't be stable.
      They become meaningful as you add more match analyses.

EXTENSIBILITY:
    - Add calibration curve plotting (requires matplotlib)
    - Add Brier score (measures probability quality, not just accuracy)
    - Add per-team breakdown (which teams are hardest to predict?)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATASET_FILE = BASE_DIR / "data" / "datasets" / "training_data.csv"
MODEL_FILE = BASE_DIR / "data" / "models" / "football_model.pkl"
FEATURES_FILE = BASE_DIR / "data" / "models" / "feature_columns.json"
PLOTS_DIR = BASE_DIR / "data" / "models"

# Human-readable labels for the 3 outcomes
LABEL_NAMES = {0: "Away Win", 1: "Draw", 2: "Home Win"}


def main():
    print("=" * 60)
    print("EVALUATE MODEL")
    print("=" * 60)

    # --- Load model and data ---
    if not MODEL_FILE.exists():
        print(f"[ERROR] Model not found: {MODEL_FILE}")
        print("Run train_model.py first.")
        return

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATASET_FILE)

    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    X = df[feature_cols].values
    y_true = df["result"].values

    # --- Predictions ---
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # --- 1. Overall Accuracy ---
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc:.3f}  ({sum(y_true == y_pred)}/{len(y_true)} correct)")

    # --- 2. Classification Report ---
    # Map integer labels to human names for readability
    target_names = [LABEL_NAMES.get(i, str(i)) for i in sorted(np.unique(y_true))]
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # --- 3. Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    print("--- Confusion Matrix ---")
    print(f"  (rows = actual, columns = predicted)")
    present_labels = sorted(np.unique(y_true))
    header = "           " + "  ".join(f"{LABEL_NAMES.get(l, str(l)):>9}" for l in present_labels)
    print(header)
    for i, row in enumerate(cm):
        row_label = LABEL_NAMES.get(present_labels[i], str(present_labels[i]))
        row_str = "  ".join(f"{v:>9}" for v in row)
        print(f"  {row_label:>9} {row_str}")
    print()

    # --- 4. Probability Distribution ---
    print("--- Prediction Probabilities (first 10 matches) ---")
    print(f"  {'Match':<30} {'Home%':>6} {'Draw%':>6} {'Away%':>6}  {'Pred':>9}  {'Actual':>9}")
    print(f"  {'-'*30} {'-----':>6} {'-----':>6} {'-----':>6}  {'-'*9}  {'-'*9}")

    for i in range(min(10, len(df))):
        home = df.iloc[i]["home_team"]
        away = df.iloc[i]["away_team"]
        match_label = f"{home} vs {away}"

        # Probabilities — model.classes_ tells us which index = which class
        proba = y_proba[i]
        classes = model.classes_
        prob_dict = {int(classes[j]): proba[j] for j in range(len(classes))}

        home_pct = prob_dict.get(2, 0) * 100
        draw_pct = prob_dict.get(1, 0) * 100
        away_pct = prob_dict.get(0, 0) * 100

        pred_label = LABEL_NAMES.get(y_pred[i], "?")
        true_label = LABEL_NAMES.get(y_true[i], "?")

        print(f"  {match_label:<30} {home_pct:>5.1f}% {draw_pct:>5.1f}% {away_pct:>5.1f}%  {pred_label:>9}  {true_label:>9}")

    # --- 5. Feature Importance (try to plot) ---
    try:
        save_feature_importance_plot(model, feature_cols)
    except Exception as e:
        print(f"\n[NOTE] Could not generate importance plot: {e}")

    print("\nEvaluation complete.")


def save_feature_importance_plot(model, feature_cols):
    """
    Attempt to extract and plot feature importances.

    CalibratedClassifierCV wraps the base estimator, so we need
    to dig into it to get .feature_importances_.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
    import matplotlib.pyplot as plt

    # Extract the base RandomForest from inside the calibrated wrapper
    if hasattr(model, "estimator"):
        # scikit-learn >= 1.2
        base = model.estimator
    elif hasattr(model, "base_estimator"):
        # older scikit-learn
        base = model.base_estimator
    elif hasattr(model, "feature_importances_"):
        base = model
    else:
        print("\n[NOTE] Model type doesn't expose feature importances directly.")
        return

    if not hasattr(base, "feature_importances_"):
        print("\n[NOTE] Base model has no feature_importances_ attribute.")
        return

    importances = base.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_cols) * 0.4)))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="#4C72B0")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance — Which tactical ratings matter most?")
    plt.tight_layout()

    plot_path = PLOTS_DIR / "feature_importance.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nFeature importance plot saved to {plot_path}")


if __name__ == "__main__":
    main()
