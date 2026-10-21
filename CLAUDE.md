# EPL match prediction

Use the terms in `CONTEXT.md` (Match, Pre-match Feature, Analyst Adjustment, ...) in code, comments and conversation. When the meaning of one changes, update the glossary. Read `docs/adr/` before changing how Match Notes, Team Profiles or Analyst Adjustments feed the model.

## The Kickoff rule

Every Pre-match Feature for a Match is computed only from Matches that finished before its Kickoff. Matches on the same date are treated as simultaneous, so they never see each other's results. Each feature change ships with a test that changes Match N's Result and asserts Match N's features are unchanged.

## Commands

Windows, using the venv at `.venv/`:

```
.venv/Scripts/python src/fetch_data.py   # season CSVs -> data/raw/
.venv/Scripts/python src/build_db.py     # data/raw/ -> data/epl.db (full rebuild)
.venv/Scripts/python src/build_dataset.py  # matches -> match_features table
.venv/Scripts/python src/train_model.py  # choose on validation seasons, refit, save
.venv/Scripts/python src/evaluate.py     # test seasons -> results/
.venv/Scripts/python src/train_model.py --live  # refit on all completed seasons for predict.py
.venv/Scripts/python -m pytest -q
```

`src/tune_elo.py` (about 7 minutes) chose the Elo constants in `ratings.py`; rerun it only when features change. `evaluate.py` is the single look at the test seasons. Make model choices with `train_model.py`'s validation table, never with test results.

`data/raw/` is committed so results are reproducible. `data/epl.db` is gitignored and rebuilt from it.

## Predictions

`src/predict.py fixtures` (every upcoming EPL fixture from football-data.co.uk) or `src/predict.py match HOME AWAY` predicts upcoming Matches, applies Analyst Adjustments from `adjustments.yaml`, and logs to `data/predictions.db`. `src/predict.py report` scores the log. The log is a record made before Kickoff, so it is committed and never rebuilt. Use `--dry-run` for any prediction that isn't a real forecast.
