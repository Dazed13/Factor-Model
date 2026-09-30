"""Unit tests for Bhavcopy × shares market-equity construction (no network)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.data.market_equity import (
    market_equity_asof,
    merge_me_into_fundamentals,
    month_end_asof_frame,
)


def _closes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [
                date(2020, 3, 30),
                date(2020, 3, 31),
                date(2020, 4, 1),
                date(2020, 6, 30),
            ],
            "symbol": ["AAA", "AAA", "AAA", "AAA"],
            "close": [100.0, 110.0, 105.0, 120.0],
        }
    )


def _shares() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "as_of_date": [date(2020, 1, 15), date(2020, 3, 31), date(2020, 6, 1)],
            "symbol": ["AAA", "AAA", "AAA"],
            "shares": [1_000.0, 2_000.0, 2_000.0],
        }
    )


class TestMarketEquityAsOf:
    def test_uses_last_close_and_shares_on_or_before(self) -> None:
        as_of = pl.DataFrame(
            {
                "as_of_date": [date(2020, 3, 31), date(2020, 6, 30)],
                "symbol": ["AAA", "AAA"],
            }
        )
        me = market_equity_asof(as_of, _closes(), _shares())
        r0 = me.filter(pl.col("as_of_date") == date(2020, 3, 31))
        assert r0["close"][0] == pytest.approx(110.0)
        assert r0["shares"][0] == pytest.approx(2_000.0)
        assert r0["market_cap"][0] == pytest.approx(220_000.0)
        assert r0["price_date"][0] == date(2020, 3, 31)
        assert r0["shares_date"][0] == date(2020, 3, 31)

        r1 = me.filter(pl.col("as_of_date") == date(2020, 6, 30))
        assert r1["market_cap"][0] == pytest.approx(120.0 * 2_000.0)

    def test_weekend_asof_uses_prior_close(self) -> None:
        as_of = pl.DataFrame(
            {"as_of_date": [date(2020, 4, 4)], "symbol": ["AAA"]}  # Sat after Apr 1
        )
        me = market_equity_asof(as_of, _closes(), _shares())
        assert me["price_date"][0] == date(2020, 4, 1)
        assert me["market_cap"][0] == pytest.approx(105.0 * 2_000.0)

    def test_missing_shares_yields_null_me(self) -> None:
        as_of = pl.DataFrame(
            {"as_of_date": [date(2019, 6, 30)], "symbol": ["AAA"]}
        )
        me = market_equity_asof(as_of, _closes(), _shares())
        assert me["market_cap"][0] is None


class TestMonthEnd:
    def test_month_end_keys(self) -> None:
        ends = month_end_asof_frame(_closes())
        assert date(2020, 3, 31) in ends["as_of_date"].to_list()
        assert date(2020, 4, 1) in ends["as_of_date"].to_list()
        assert date(2020, 6, 30) in ends["as_of_date"].to_list()


class TestMergeMe:
    def test_overlay_recomputes_btm(self) -> None:
        book = pl.DataFrame(
            {
                "as_of_date": [date(2020, 3, 31)],
                "available_date": [date(2020, 6, 30)],
                "symbol": ["AAA"],
                "market_cap": [None],
                "book_value": [50_000.0],
                "book_to_market": [None],
            }
        )
        me = pl.DataFrame(
            {
                "as_of_date": [date(2020, 3, 31)],
                "symbol": ["AAA"],
                "close": [110.0],
                "shares": [2_000.0],
                "market_cap": [220_000.0],
                "price_date": [date(2020, 3, 31)],
                "shares_date": [date(2020, 3, 31)],
            }
        )
        merged = merge_me_into_fundamentals(book, me)
        assert merged.height == 1
        assert merged["market_cap"][0] == pytest.approx(220_000.0)
        assert merged["book_value"][0] == pytest.approx(50_000.0)
        assert merged["book_to_market"][0] == pytest.approx(50_000.0 / 220_000.0)
        assert merged["available_date"][0] == date(2020, 6, 30)
