"""Score the saved model on the test seasons, against every Baseline.

Run once, after train_model.py has picked and refit the model. Writes
results/test_metrics.json plus two charts in results/.
"""

import json
import sqlite3
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

from build_dataset import DB_PATH
from modeling import (TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS, base_rates, home_win_always, load_dataset,
                      market_probabilities, predict, score, split)
from train_model import MODEL_PATH, SELECTION_PATH

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Chart tokens (light surface, validated categorical slots 1 and 2).
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLOR, MARKET_COLOR, BASELINE_COLOR = "#2a78d6", "#eb6834", "#b5b4ae"


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        df = load_dataset(conn)
    history = split(df, TRAIN_SEASONS + VALIDATION_SEASONS)
    test = split(df, TEST_SEASONS)
    assert test[["market_avg_close_home", "market_avg_close_draw", "market_avg_close_away"]].notna().all().all()

    model = joblib.load(MODEL_PATH)
    model_name = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["selected"]

    forecasts = {
        "Home win always": home_win_always(len(test)),
        "Base Rates": base_rates(history, len(test)),
        model_name: predict(model, test),
        "Market Probabilities": market_probabilities(test),
    }
    table = pd.DataFrame([{"forecast": name, **score(test["result"], p)} for name, p in forecasts.items()])
    print(f"Test seasons: {', '.join(TEST_SEASONS)} ({len(test)} matches)\n")
    print(table.to_string(index=False, float_format="%.4f"))

    base = table.set_index("forecast").loc["Base Rates", "log_loss"]
    ours = table.set_index("forecast").loc[model_name, "log_loss"]
    market = table.set_index("forecast").loc["Market Probabilities", "log_loss"]
    print(f"\nLog loss vs Base Rates: {ours - base:+.4f} ({(ours - base) / base:+.1%})")
    print(f"Log loss vs market:     {ours - market:+.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps({
        "test_seasons": TEST_SEASONS,
        "matches": len(test),
        "model": model_name,
        # Log loss is infinite for "Home win always" (it gives 0% to Results that happen).
        "scores": json.loads(table.replace([np.inf], None).to_json(orient="records")),
    }, indent=2), encoding="utf-8")

    plot_log_loss(table[table["forecast"] != "Home win always"], model_name)
    plot_calibration(test, forecasts[model_name], forecasts["Market Probabilities"], model_name)
    print(f"\nWrote {RESULTS_DIR / 'test_metrics.json'} and charts to {RESULTS_DIR}")


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=10)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def plot_log_loss(table: pd.DataFrame, model_name: str) -> None:
    colors = {model_name: MODEL_COLOR, "Market Probabilities": MARKET_COLOR}
    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    labels = list(table["forecast"])
    bars = ax.barh(labels, table["log_loss"], height=0.55,
                   color=[colors.get(f, BASELINE_COLOR) for f in labels])
    for bar, value in zip(bars, table["log_loss"]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Log loss on 2024-25 and 2025-26 (lower is better)", color=INK_MUTED)
    ax.set_title("Test-season log loss", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "log_loss.png", dpi=150)
    plt.close(fig)


def calibration_points(results: pd.Series, p_home: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed home-win rate in equal-sized probability bins."""
    df = pd.DataFrame({"p": p_home, "home_win": (results.to_numpy() == "H").astype(float)})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    return df.groupby("bin").agg(predicted=("p", "mean"), observed=("home_win", "mean"))


def plot_calibration(test: pd.DataFrame, model_p: np.ndarray, market_p: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=SURFACE)
    style(ax)
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, label="Perfectly calibrated")
    for name, p, color in [(model_name, model_p, MODEL_COLOR), ("Market Probabilities", market_p, MARKET_COLOR)]:
        pts = calibration_points(test["result"], p[:, 2])
        ax.plot(pts["predicted"], pts["observed"], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted home-win probability", color=INK_MUTED)
    ax.set_ylabel("Actual home-win rate", color=INK_MUTED)
    ax.set_title("Calibration on test seasons (10 equal-sized bins)", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    legend = ax.legend(loc="upper left", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "calibration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
