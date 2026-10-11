"""Download EPL season CSVs from football-data.co.uk into data/raw/.

One file per season, named by its start year, e.g. data/raw/E0_2005-06.csv.
Completed seasons are skipped if already present; the in-progress season is
always re-downloaded so new results are picked up.
"""

import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

FIRST_SEASON = 2005
CURRENT_SEASON = 2026  # 2026-27, in progress


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(FIRST_SEASON, CURRENT_SEASON + 1):
        path = RAW_DIR / f"E0_{season_label(year)}.csv"
        if path.exists() and year < CURRENT_SEASON:
            continue
        url = URL.format(code=season_code(year))
        print(f"Downloading {season_label(year)} from {url}")
        with urllib.request.urlopen(url, timeout=30) as resp:
            path.write_bytes(resp.read())
    print(f"Season files in {RAW_DIR}: {len(list(RAW_DIR.glob('E0_*.csv')))}")


if __name__ == "__main__":
    sys.exit(main())
