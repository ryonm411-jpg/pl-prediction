# EPL Match Prediction

Estimates Home/Draw/Away probabilities for English Premier League matches from information available before kickoff, and compares them against simple baselines and the betting market.

## Language

### Matches and outcomes

**Match**:
A single EPL fixture between a home team and an away team, identified by season, date and the two teams.
_Avoid_: Game, fixture (when meaning a played match)

**Result**:
The full-time outcome of a Match from the home team's perspective: Home win, Draw or Away win.
_Avoid_: Outcome, label, FTR

**Kickoff**:
The moment that separates what may be known about a Match (before) from what may not (after).

### Features

**Pre-match Feature**:
A number describing a Match that is computed only from information available before Kickoff, such as results and stats from earlier Matches.
_Avoid_: Feature (unqualified), input

**In-match Stat**:
A statistic recorded during a Match (shots, corners, cards). It may be used only as an input to Pre-match Features of later Matches, never for the Match it came from.
_Avoid_: Match stat, feature

**Elo Rating**:
A team's strength score, updated after each Match in date order, so the value used for a Match reflects only earlier Results.
_Avoid_: Rating (unqualified), power ranking

### Analyst layer

**Match Note**:
A hand-written tactical analysis of one Match, written after watching it. It includes subjective Tactical Ratings.
_Avoid_: Match analysis, notes file

**Tactical Rating**:
A 0–100 subjective score for one team's attribute (e.g. Pressing Intensity) in one Match Note. It is an observation made after Kickoff, so it is never a Pre-match Feature for that Match.
_Avoid_: Rating (unqualified), score

**Team Profile**:
A hand-written description of a team's squad and style for the current season only. It is used to explain predictions for upcoming Matches, not to train the model.
_Avoid_: Team file, team notes

**Analyst Adjustment**:
A saved tactical claim, with a written reason, that shifts one team's Elo Rating by a size the analyst chooses. It applies to Matches matching its scope (a team pairing plus required players, or any opponent of one team), is confirmed per Match, and expires at season end unless renewed.
_Avoid_: Edge, tweak, override, manual rating

**Prediction**:
A record made before Kickoff of the model's probabilities, the adjusted probabilities, the Analyst Adjustments applied and, optionally, the Market Probabilities for one Match. It is scored once the Result is known.
_Avoid_: Forecast, pick, bet

### Evaluation

**Baseline**:
A deliberately simple probability forecast that the model must beat to be worth anything. The project uses Home-win-always, Base Rates and Market Probabilities.

**Base Rates**:
The long-run Home/Draw/Away frequencies from training seasons, predicted identically for every Match.
_Avoid_: Prior, naive model

**Market Probabilities**:
Bookmaker closing odds converted to Home/Draw/Away probabilities, with the bookmaker margin removed. This is the strongest Baseline and is not expected to be beaten.
_Avoid_: Odds (when meaning probabilities), implied odds
