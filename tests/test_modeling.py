import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from modeling import (TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS, base_rates,  # noqa: E402
                      market_probabilities, score)


def test_market_probabilities_remove_the_margin():
    odds = pd.DataFrame({"market_avg_close_away": [4.0, 1.5],
                         "market_avg_close_draw": [3.6, 4.2],
                         "market_avg_close_home": [1.9, 6.5]})
    implied = 1 / odds.to_numpy()
    assert (implied.sum(axis=1) > 1).all()  # bookmakers' margin

    p = market_probabilities(odds)
    assert p.sum(axis=1) == pytest.approx([1.0, 1.0])
    # Columns come back in CLASSES order: away, draw, home.
    assert p[0].argmax() == 2 and p[1].argmax() == 0


def test_score_on_a_hand_worked_example():
    results = pd.Series(["H", "D"])
    proba = np.array([[0.2, 0.3, 0.5],
                      [0.25, 0.5, 0.25]])
    s = score(results, proba)
    assert s["accuracy"] == 1.0
    assert s["log_loss"] == pytest.approx(-(math.log(0.5) + math.log(0.5)) / 2)
    # Match 1: 0.2² + 0.3² + 0.5² = 0.38; Match 2: 0.25² + 0.5² + 0.25² = 0.375.
    assert s["brier"] == pytest.approx((0.38 + 0.375) / 2)


def test_certain_wrong_forecast_has_infinite_log_loss():
    s = score(pd.Series(["A"]), np.array([[0.0, 0.0, 1.0]]))
    assert s["log_loss"] == math.inf


def test_base_rates_match_training_frequencies():
    train = pd.DataFrame({"result": ["H", "H", "D", "A"]})
    assert base_rates(train, 2) == pytest.approx(np.array([[0.25, 0.25, 0.5]] * 2))


def test_seasons_are_disjoint_and_in_time_order():
    assert not set(TRAIN_SEASONS) & set(VALIDATION_SEASONS)
    assert not set(VALIDATION_SEASONS) & set(TEST_SEASONS)
    assert max(TRAIN_SEASONS) < min(VALIDATION_SEASONS) <= max(VALIDATION_SEASONS) < min(TEST_SEASONS)
