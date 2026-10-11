# Tactical notes explain predictions; they do not feed the model

Match Notes record Tactical Ratings observed while watching the Match, so using them as features for that same Match leaks the Result into the inputs. Team Profiles describe only the current season, so they can't be applied to historical Matches. We therefore train only on Pre-match Features derived from historical football-data.co.uk results (2005-06 to 2025-26). Match Notes and Team Profiles become an analyst layer, shown next to the model's probabilities for upcoming Matches.

## Considered Options

- **Rolling averages of earlier Match Notes as Pre-match Features:** this is correct, but there will only be a few hand-written notes, so the feature would be missing for almost every Match.
- **Drop the notes entirely:** this throws away the project's most distinctive part.
