-- One row per played EPL Match, loaded from football-data.co.uk season CSVs.
-- Team names use football-data.co.uk spellings (e.g. 'Man City', 'Nott''m Forest').
-- Every stat column is an In-match Stat: it describes this Match only and must
-- never be used as a Pre-match Feature for the same Match (see CONTEXT.md).

DROP TABLE IF EXISTS matches;

CREATE TABLE matches (
    match_id               INTEGER PRIMARY KEY,
    season                 TEXT    NOT NULL,          -- '2025-26'
    match_date             TEXT    NOT NULL,          -- ISO 'YYYY-MM-DD'
    kickoff_time           TEXT,                      -- 'HH:MM', from 2019-20 only
    home_team              TEXT    NOT NULL,
    away_team              TEXT    NOT NULL,
    home_goals             INTEGER NOT NULL,
    away_goals             INTEGER NOT NULL,
    result                 TEXT    NOT NULL CHECK (result IN ('H', 'D', 'A')),

    home_shots             INTEGER,
    away_shots             INTEGER,
    home_shots_on_target   INTEGER,
    away_shots_on_target   INTEGER,
    home_corners           INTEGER,
    away_corners           INTEGER,
    home_yellows           INTEGER,
    away_yellows           INTEGER,
    home_reds              INTEGER,
    away_reds              INTEGER,

    -- Decimal odds. Closing odds are the last price before Kickoff.
    b365_home              REAL,
    b365_draw              REAL,
    b365_away              REAL,
    pinnacle_close_home    REAL,                      -- 2012-13 to Jan 2026
    pinnacle_close_draw    REAL,
    pinnacle_close_away    REAL,
    market_avg_close_home  REAL,                      -- 2019-20 onwards
    market_avg_close_draw  REAL,
    market_avg_close_away  REAL,

    UNIQUE (season, home_team, away_team),
    CHECK (home_team <> away_team)
);

CREATE INDEX idx_matches_date ON matches (match_date, kickoff_time);
CREATE INDEX idx_matches_home ON matches (home_team, match_date);
CREATE INDEX idx_matches_away ON matches (away_team, match_date);
