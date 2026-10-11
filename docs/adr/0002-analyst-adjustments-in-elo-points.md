# Analyst Adjustments are hand-set Elo shifts applied after training

Analyst Adjustments shift a team's Elo Rating at prediction time by an amount the analyst chooses. They are never training inputs, because no historical record of them exists. Expressing them in Elo points keeps the adjusted probabilities consistent, so they still sum to 1 and behave sensibly near certainty. Every adjusted Prediction is logged before Kickoff, so the adjustments' value can be measured against model-only probabilities.

## Considered Options

- **Percentage-point shifts on probabilities:** these don't say which other outcome the probability comes from, and they break near 0% and 100%.
- **Deriving the size from Team Profile player ratings:** this was rejected. Those ratings are subjective, so turning their differences into odds would look precise without being precise.
