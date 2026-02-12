# Football Match Prediction Model

## Project Purpose
To build a probabilistic football match prediction system that bridges the gap between qualitative expert analysis and quantitative modeling. This project converts human-written tactical notes into structured numerical features to train interpretable machine learning models.

## Data Flow
1.  **Input:** Analyst writes Markdown files for matches (`match_analysis/`) and team profiles (`teams/`).
2.  **Parsing:** `parse_notes.py` converts Markdown into structured JSON (`data/raw_matches/`).
3.  **Processing:** `build_dataset.py` merges match data with team profiles to create feature vectors (`data/datasets/`).
4.  **Modeling:** `train_model.py` trains a probabilistic classifier to predict match outcomes (Home/Draw/Away).
5.  **Output:** `evaluate.py` validates model performance, and `predict.py` generates probabilities for future games.

## High-Level Model Logic
The model avoids "black box" deep learning in favor of explainability. It relies on:
*   **Tactical Ratings:** 0-100 scores for specific team attributes (e.g., "Pressing Intensity", "Counter-Attack Threat").
*   **Formations:** Structured data on how teams set up (starting, on-ball, off-ball).
*   **Deltas:** The difference between Home and Away ratings (e.g., Home Attack vs. Away Defense).
*   **Probabilistic Output:** Calibrated probabilities for Home Win, Draw, and Away Win, rather than just a single class prediction.
