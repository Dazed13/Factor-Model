"""Unit tests for Screener.in book-equity parsing (no network)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.data.fundamentals import apply_reporting_lag
from src.data.screener import (
    CRORE_TO_INR,
    book_equity_crore_from_balance_sheet,
    merge_book_into_fundamentals,
    parse_balance_sheet_table,
    parse_screener_number,
    parse_screener_period,
)

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "screener_reliance_balance_sheet.html"
)


@pytest.fixture
def reliance_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


class TestParsers:
    def test_period_month_end(self) -> None:
        assert parse_screener_period("Mar 2020") == date(2020, 3, 31)
        assert parse_screener_period("Sep 2025") == date(2025, 9, 30)
        assert parse_screener_period("TTM") is None
        assert parse_screener_period("") is None

    def test_number_commas(self) -> None:
        assert parse_screener_number("13,532") == pytest.approx(13532.0)
        assert parse_screener_number("-") is None
        assert parse_screener_number("") is None


class TestBalanceSheetFixture:
    def test_equity_and_reserves_present(self, reliance_html: str) -> None:
        table = parse_balance_sheet_table(reliance_html)
        assert "Equity Capital" in table
        assert "Reserves" in table
        mar2020 = date(2020, 3, 31)
        assert table["Equity Capital"][mar2020] == pytest.approx(6339.0)
        assert table["Reserves"][mar2020] == pytest.approx(442827.0)

    def test_book_equity_inr_and_year_filter(self, reliance_html: str) -> None:
        table = parse_balance_sheet_table(reliance_html)
        book_cr = book_equity_crore_from_balance_sheet(table)
        mar2020 = date(2020, 3, 31)
        assert book_cr[mar2020] == pytest.approx(6339.0 + 442827.0)

        rows = [
            {
                "as_of_date": d,
                "symbol": "RELIANCE",
                "market_cap": None,
                "book_value": v * CRORE_TO_INR,
                "book_to_market": None,
            }
            for d, v in book_cr.items()
            if 2020 <= d.year <= 2025
        ]
        df = apply_reporting_lag(pl.DataFrame(rows), lag_months=3)
        years = set(df.get_column("as_of_date").dt.year().to_list())
        assert years <= set(range(2020, 2026))
        assert date(2020, 3, 31) in df.get_column("as_of_date").to_list()
        # Reporting lag: Mar 2020 → Jun 2020
        row = df.filter(pl.col("as_of_date") == date(2020, 3, 31))
        assert row["available_date"][0] == date(2020, 6, 30)
        assert row["book_value"][0] == pytest.approx((6339.0 + 442827.0) * CRORE_TO_INR)


class TestMergeBook:
    def test_overlay_preserves_me_and_recomputes_btm(self) -> None:
        base = pl.DataFrame(
            {
                "as_of_date": [date(2020, 3, 31)],
                "available_date": [date(2020, 6, 30)],
                "symbol": ["RELIANCE"],
                "market_cap": [1.0e12],
                "book_value": [1.0],
                "book_to_market": [1.0e-12],
            }
        )
        book = pl.DataFrame(
            {
                "as_of_date": [date(2020, 3, 31), date(2021, 3, 31)],
                "available_date": [date(2020, 6, 30), date(2021, 6, 30)],
                "symbol": ["RELIANCE", "RELIANCE"],
                "market_cap": [None, None],
                "book_value": [3.91e12, 4.0e12],
                "book_to_market": [None, None],
            }
        )
        merged = merge_book_into_fundamentals(base, book)
        assert merged.height == 2
        r20 = merged.filter(pl.col("as_of_date") == date(2020, 3, 31))
        assert r20["market_cap"][0] == pytest.approx(1.0e12)
        assert r20["book_value"][0] == pytest.approx(3.91e12)
        assert r20["book_to_market"][0] == pytest.approx(3.91)
        r21 = merged.filter(pl.col("as_of_date") == date(2021, 3, 31))
        assert r21["book_value"][0] == pytest.approx(4.0e12)
        assert r21["market_cap"][0] is None
