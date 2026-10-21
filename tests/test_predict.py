import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import predict  # noqa: E402

FIXTURES = """\
Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA
E0,24/10/2026,12:30,Man City,Liverpool,2.10,3.60,3.40
E1,24/10/2026,15:00,Leicester,Derby,2.0,3.4,3.8
SP1,25/10/2026,20:00,Real Madrid,Barcelona,2.2,3.7,3.1
"""


def test_upcoming_fixtures_keep_only_the_premier_league(tmp_path):
    path = tmp_path / "fixtures.csv"
    path.write_text(FIXTURES, encoding="utf-8-sig")  # the real file starts with a byte-order mark
    fixtures = predict.upcoming_epl_fixtures(path.as_uri())
    assert list(zip(fixtures["HomeTeam"], fixtures["AwayTeam"])) == [("Man City", "Liverpool")]
    assert fixtures.iloc[0]["AvgH"] == 2.10
