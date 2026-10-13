"""Choose Elo's K factor and home advantage by validation log loss.

For each setting, rebuild every Pre-match Feature, fit logistic regression on
the training seasons and score it on the validation seasons. Test seasons are
never touched. Copy the winning values into ratings.py, then rerun
build_dataset.py.
"""

import sqlite3
import sys

import pandas as pd

from build_dataset import DB_PATH, compute_features
from modeling import TRAIN_SEASONS, VALIDATION_SEASONS, candidate_models, fit, predict, score
from ratings import EloTable

K_FACTORS = [10, 15, 20, 25, 30, 40]
HOME_ADVANTAGES = [30, 45, 60, 75, 90]


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        matches = pd.read_sql("SELECT * FROM matches", conn)
    info = matches[["match_id", "season", "result"]]

    results = []
    for k in K_FACTORS:
        for home in HOME_ADVANTAGES:
            features = compute_features(matches, EloTable(k_factor=k, home_advantage=home))
            df = info.merge(features, on="match_id")
            train = df[df["season"].isin(TRAIN_SEASONS)]
            validation = df[df["season"].isin(VALIDATION_SEASONS)]
            model = fit(candidate_models()["Logistic regression"], train)
            results.append({"k_factor": k, "home_advantage": home,
                            **score(validation["result"], predict(model, validation))})

    table = pd.DataFrame(results).sort_values("log_loss")
    print(table.head(10).to_string(index=False, float_format="%.4f"))
    best = table.iloc[0]
    print(f"\nBest: K_FACTOR = {best.k_factor:g}, HOME_ADVANTAGE = {best.home_advantage:g}")


if __name__ == "__main__":
    sys.exit(main())
