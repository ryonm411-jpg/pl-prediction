"""Shared pieces for training and evaluation: the time split, candidate models,
Baselines and probability metrics.

Every probability array in this project has columns in CLASSES order:
Away win, Draw, Home win (sklearn sorts the string labels this way).
"""

import sqlite3

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_dataset import FEATURE_COLUMNS

CLASSES = np.array(["A", "D", "H"])

# 2005-06 only warms up Elo and the rolling windows: every team starts it at
# the same rating, so its features say little. 2026-27 is in progress.
TRAIN_SEASONS = [f"{y}-{(y + 1) % 100:02d}" for y in range(2006, 2022)]
VALIDATION_SEASONS = ["2022-23", "2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

DATASET_QUERY = """
SELECT m.match_id, m.season, m.match_date, m.home_team, m.away_team, m.result,
       m.market_avg_close_home, m.market_avg_close_draw, m.market_avg_close_away,
       f.*
FROM matches m
JOIN match_features f USING (match_id)
ORDER BY m.match_date, m.match_id
"""


def load_dataset(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql(DATASET_QUERY, conn)
    return df.loc[:, ~df.columns.duplicated()]


def split(df: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    return df[df["season"].isin(seasons)]


def candidate_models() -> dict:
    """The models compared on the validation seasons.

    Calibration uses time-ordered folds, so each calibration fold is always
    later than the data the fold's model was trained on.
    """
    def logistic():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    def forest():
        # Large leaves keep the leaf frequencies (the probabilities) stable.
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=50, max_features="sqrt",
                                      random_state=42, n_jobs=-1)

    def calibrated(model):
        return CalibratedClassifierCV(model, method="sigmoid", cv=TimeSeriesSplit(n_splits=5))

    return {
        "Logistic regression": logistic(),
        "Logistic regression + calibration": calibrated(logistic()),
        "Random forest": forest(),
        "Random forest + calibration": calibrated(forest()),
    }


def fit(model, df: pd.DataFrame):
    return model.fit(df[FEATURE_COLUMNS], df["result"])


def predict(model, df: pd.DataFrame) -> np.ndarray:
    assert list(model.classes_) == list(CLASSES)
    return model.predict_proba(df[FEATURE_COLUMNS])


# --- Baselines --------------------------------------------------------------

def home_win_always(n: int) -> np.ndarray:
    return np.tile([0.0, 0.0, 1.0], (n, 1))


def base_rates(train: pd.DataFrame, n: int) -> np.ndarray:
    rates = train["result"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
    return np.tile(rates, (n, 1))


def market_probabilities(df: pd.DataFrame) -> np.ndarray:
    """Average closing odds -> probabilities, removing the bookmaker margin by
    scaling the three implied probabilities so they sum to 1."""
    odds = df[["market_avg_close_away", "market_avg_close_draw", "market_avg_close_home"]].to_numpy()
    implied = 1.0 / odds
    return implied / implied.sum(axis=1, keepdims=True)


# --- Metrics ----------------------------------------------------------------

def score(results: pd.Series, proba: np.ndarray) -> dict[str, float]:
    """Accuracy, log loss and Brier score (lower is better for the last two).

    Brier is the multi-class form: the squared distance between the probability
    vector and the one-hot Result, summed over the three outcomes.
    """
    onehot = (results.to_numpy()[:, None] == CLASSES[None, :]).astype(float)
    p_actual = (proba * onehot).sum(axis=1)
    with np.errstate(divide="ignore"):
        log_loss = float(-np.log(p_actual).mean())
    return {
        "accuracy": float((proba.argmax(axis=1) == onehot.argmax(axis=1)).mean()),
        "log_loss": log_loss,
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
    }
