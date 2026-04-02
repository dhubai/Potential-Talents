from __future__ import annotations

from talentfit.ingest import load_candidates, parse_connections


def test_parse_connections_basic() -> None:
    assert parse_connections("500+") == 500
    assert parse_connections("250") == 250
    assert parse_connections(123) == 123
    assert parse_connections("") is None
    assert parse_connections(None) is None


def test_load_candidates_xlsx_smoke() -> None:
    res = load_candidates("potential-talents.xlsx")
    df = res.df
    assert len(df) == 104
    assert {"id", "job_title", "location", "connections"}.issubset(df.columns)
    assert df["id"].isna().sum() == 0
    assert df["job_title"].isna().sum() == 0
    assert df["connections"].isna().sum() == 0

