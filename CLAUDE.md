# EPL match prediction

Use the terms in `CONTEXT.md` (Match, Pre-match Feature, Analyst Adjustment, ...) in code, comments and conversation. When the meaning of one changes, update the glossary. Read `docs/adr/` before changing how Match Notes, Team Profiles or Analyst Adjustments feed the model.

## The Kickoff rule

Every Pre-match Feature for a Match is computed only from Matches that finished before its Kickoff. Matches on the same date are treated as simultaneous, so they never see each other's results. Each feature change ships with a test that changes Match N's Result and asserts Match N's features are unchanged.

## Commands

Windows, using the venv at `.venv/`:

```
.venv/Scripts/python src/fetch_data.py   # season CSVs -> data/raw/
.venv/Scripts/python src/build_db.py     # data/raw/ -> data/epl.db (full rebuild)
.venv/Scripts/python -m pytest -q
```

`data/raw/` is committed so results are reproducible. `data/epl.db` is gitignored and rebuilt from it.

## In transition

`build_dataset.py`, `train_model.py`, `evaluate.py` and `predict.py` still implement the old design, which used Tactical Ratings as features. They are being rewritten on top of `data/epl.db`. Treat them as reference, not as patterns to copy.
