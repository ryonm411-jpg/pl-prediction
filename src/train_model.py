"""Pick a model on the validation seasons, then refit it for the test run.

1. Fit each candidate model on the training seasons and score it on the
   validation seasons. Log loss decides: it rewards honest probabilities, not
   just picking the most likely Result.
2. Refit the winner on training + validation seasons and save it to
   data/models/model.joblib. src/evaluate.py scores it on the test seasons once.

The test seasons are never loaded here, except by `--live`: after evaluate.py
has scored the model, `train_model.py --live` refits the selected model on every
completed season and saves data/models/model_live.joblib for predict.py. The
evaluated model is kept unchanged, as the evidence behind the README's results.
"""

import json
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd

from build_dataset import DB_PATH, FEATURE_COLUMNS
from modeling import (COMPLETED_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS, base_rates, candidate_models, fit, load_dataset,
                      predict, score, split)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
SELECTION_PATH = MODEL_DIR / "selection.json"
LIVE_MODEL_PATH = MODEL_DIR / "model_live.joblib"


def main() -> None:
    if "--live" in sys.argv[1:]:
        fit_live()
        return
    with sqlite3.connect(DB_PATH) as conn:
        df = load_dataset(conn)
    train, validation = split(df, TRAIN_SEASONS), split(df, VALIDATION_SEASONS)
    print(f"Train: {len(train)} matches ({TRAIN_SEASONS[0]} to {TRAIN_SEASONS[-1]})")
    print(f"Validation: {len(validation)} matches ({', '.join(VALIDATION_SEASONS)})\n")

    rows = [{"model": "Base Rates (baseline)", **score(validation["result"], base_rates(train, len(validation)))}]
    for name, model in candidate_models().items():
        fit(model, train)
        rows.append({"model": name, **score(validation["result"], predict(model, validation))})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format="%.4f"))

    candidates = table[~table["model"].str.contains("baseline")]
    best = candidates.sort_values("log_loss").iloc[0]["model"]
    print(f"\nSelected: {best}")

    final = fit(candidate_models()[best], pd.concat([train, validation]))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    SELECTION_PATH.write_text(json.dumps({
        "selected": best,
        "trained_on": TRAIN_SEASONS + VALIDATION_SEASONS,
        "features": FEATURE_COLUMNS,
        "validation": rows,
    }, indent=2), encoding="utf-8")
    print(f"Refit on train + validation; saved to {MODEL_PATH}")

    if hasattr(final, "named_steps"):
        coefs = pd.DataFrame(final.named_steps["logisticregression"].coef_.T,
                             index=FEATURE_COLUMNS, columns=["away", "draw", "home"])
        print("\nStandardised coefficients (log-odds per standard deviation):")
        print(coefs.round(3).to_string())


def fit_live() -> None:
    if not SELECTION_PATH.exists():
        sys.exit("Run train_model.py (and evaluate.py) first, so a model has been selected and scored.")
    selected = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["selected"]
    with sqlite3.connect(DB_PATH) as conn:
        completed = split(load_dataset(conn), COMPLETED_SEASONS)
    joblib.dump(fit(candidate_models()[selected], completed), LIVE_MODEL_PATH)
    print(f"Refit {selected} on {len(completed)} matches ({COMPLETED_SEASONS[0]} to {COMPLETED_SEASONS[-1]}); "
          f"saved to {LIVE_MODEL_PATH}")


if __name__ == "__main__":
    sys.exit(main())
