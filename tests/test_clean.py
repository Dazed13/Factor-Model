"""Phase-2 tests: cleaning, liquidation policy, store partitions, returns pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.data.bhavcopy import ALLOWED_SERIES
from src.data.clean import (
    apply_liquidation_policy,
    clean_bhavcopy,
    deduplicate_bhavcopy,
    filter_equity_series,
)
from src.data.pipeline import PipelineConfig, join_fundamentals_asof, run_phase2_pipeline
from src.data.returns import build_returns_panel, compute_price_returns, ensure_adj_close
from src.data.risk_free import build_risk_free_frame
from src.data.store import (
    assert_snappy_parquet,
    connect,
    list_partitions,
    query_df,
    read_partitioned_parquet,
    register_parquet_view,
    write_partitioned_parquet,
)


def _raw_panel() -> pl.DataFrame:
    """Two symbols; AAA stops trading after day 2; BBB continues; BZ junk series."""
    return pl.DataFrame(
        {
            "trade_date": [
                date(2020, 1, 2),
                date(2020, 1, 3),
                date(2020, 1, 2),
                date(2020, 1, 3),
                date(2020, 1, 6),
                date(2020, 1, 2),
                date(2020, 1, 2),  # duplicate EQ/BE
            ],
            "symbol": ["AAA", "AAA", "BBB", "BBB", "BBB", "JUNK", "AAA"],
            "series": ["EQ", "EQ", "EQ", "EQ", "EQ", "BZ", "BE"],
            "open": [100.0, 110.0, 50.0, 51.0, 52.0, 1.0, 100.0],
            "high": [101.0, 111.0, 51.0, 52.0, 53.0, 1.0, 101.0],
            "low": [99.0, 109.0, 49.0, 50.0, 51.0, 1.0, 99.0],
            "close": [100.0, 110.0, 50.0, 51.0, 52.0, 1.0, 100.0],
            "last": [100.0, 110.0, 50.0, 51.0, 52.0, 1.0, 100.0],
            "prev_close": [99.0, 100.0, 49.0, 50.0, 51.0, 1.0, 99.0],
            "tottrdqty": [1e6, 1e6, 2e6, 2e6, 2e6, 10.0, 100.0],
            "tottrdval": [1e8, 1.1e8, 1e8, 1.02e8, 1.04e8, 10.0, 1e4],
            "timestamp": ["2020"] * 7,
            "adj_factor": [1.0] * 7,
        }
    )


class TestClean:
    def test_filter_equity_series(self) -> None:
        df = filter_equity_series(_raw_panel())
        assert set(df["series"].unique().to_list()) <= ALLOWED_SERIES
        assert "JUNK" not in df["symbol"].to_list()

    def test_dedupe_prefers_eq(self) -> None:
        df = deduplicate_bhavcopy(filter_equity_series(_raw_panel()))
        aaa_day1 = df.filter(
            (pl.col("symbol") == "AAA") & (pl.col("trade_date") == date(2020, 1, 2))
        )
        assert aaa_day1.height == 1
        assert aaa_day1["series"][0] == "EQ"

    def test_liquidation_retains_history_with_nan(self) -> None:
        cleaned = clean_bhavcopy(_raw_panel(), apply_calendar=False)
        calendar = [
            date(2020, 1, 2),
            date(2020, 1, 3),
            date(2020, 1, 6),
            date(2020, 1, 7),
        ]
        panel = apply_liquidation_policy(cleaned, calendar=calendar)

        # AAA history retained through calendar end
        aaa = panel.filter(pl.col("symbol") == "AAA").sort("trade_date")
        assert aaa.height == 4
        assert aaa.filter(pl.col("trade_date") == date(2020, 1, 2))["close"][0] == 100.0

        # After last trade (Jan 3), later dates are liquidated with null close
        post = aaa.filter(pl.col("trade_date") > date(2020, 1, 3))
        assert post.height == 2
        assert post["is_liquidated"].all()
        assert post["close"].null_count() == post.height

        # BBB still trading on Jan 6 — not liquidated that day
        bbb_jan6 = panel.filter(
            (pl.col("symbol") == "BBB") & (pl.col("trade_date") == date(2020, 1, 6))
        )
        assert bbb_jan6["is_liquidated"][0] is False
        assert bbb_jan6["close"][0] == pytest.approx(52.0)


class TestStore:
    def test_year_partitions_roundtrip(self, tmp_path: Path) -> None:
        df = clean_bhavcopy(_raw_panel())
        paths = write_partitioned_parquet(df, tmp_path / "bhavcopy", partition_by="year")
        assert_snappy_parquet(paths)
        assert list_partitions(tmp_path / "bhavcopy") == ["2020"]

        reloaded = read_partitioned_parquet(tmp_path / "bhavcopy", years=[2020])
        assert reloaded.height == df.height
        assert set(reloaded["symbol"].to_list()) == set(df["symbol"].to_list())

    def test_duckdb_view(self, tmp_path: Path) -> None:
        df = clean_bhavcopy(_raw_panel())
        write_partitioned_parquet(df, tmp_path / "bhavcopy", partition_by="year")
        db = tmp_path / "test.duckdb"
        con = connect(db)
        try:
            register_parquet_view(con, "bhavcopy", tmp_path / "bhavcopy", partition_by="year")
            out = query_df(
                "SELECT symbol, COUNT(*) AS n FROM bhavcopy GROUP BY 1 ORDER BY 1",
                con=con,
            )
            assert out.height >= 2
            assert "AAA" in out["symbol"].to_list()
        finally:
            con.close()


class TestReturns:
    def test_returns_require_adj_close_path(self) -> None:
        df = clean_bhavcopy(_raw_panel())
        with_adj = ensure_adj_close(df)
        assert "adj_close" in with_adj.columns
        rets = compute_price_returns(with_adj)
        aaa = rets.filter(pl.col("symbol") == "AAA").sort("trade_date")
        assert aaa["ret"][0] is None or (isinstance(aaa["ret"][0], float) and np.isnan(aaa["ret"][0]))
        assert aaa["ret"][1] == pytest.approx(0.10)

    def test_liquidated_returns_are_null(self) -> None:
        cleaned = clean_bhavcopy(_raw_panel())
        cleaned = ensure_adj_close(cleaned)
        calendar = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
        panel = apply_liquidation_policy(cleaned, calendar=calendar)
        panel = ensure_adj_close(panel)
        rets = build_returns_panel(panel, rf=None)
        aaa_liq = rets.filter(
            (pl.col("symbol") == "AAA") & (pl.col("is_liquidated") == True)  # noqa: E712
        )
        assert aaa_liq.height >= 1
        assert aaa_liq["ret"].null_count() == aaa_liq.height

    def test_excess_returns_india_rf(self) -> None:
        cleaned = ensure_adj_close(clean_bhavcopy(_raw_panel()))
        rf = build_risk_free_frame(
            [date(2020, 1, 2), date(2020, 1, 6)],
            [6.5, 6.5],
            source="rbi_91d_tbill",
        )
        out = build_returns_panel(cleaned, rf=rf)
        assert "excess_ret" in out.columns
        assert out["rf_source"].drop_nulls().unique().to_list() == ["rbi_91d_tbill"]


class TestPipeline:
    def test_run_phase2_idempotent(self, tmp_path: Path) -> None:
        cfg = PipelineConfig(
            processed_bhavcopy_dir=tmp_path / "bhavcopy",
            processed_returns_dir=tmp_path / "returns",
            processed_fundamentals_dir=tmp_path / "fundamentals",
            duckdb_path=tmp_path / "fm.duckdb",
            apply_calendar=True,
        )
        raw = _raw_panel()
        r1 = run_phase2_pipeline(cfg, bhavcopy=raw, register_duckdb=True)
        r2 = run_phase2_pipeline(cfg, bhavcopy=raw, register_duckdb=True)

        assert r1.clean_prices.height == r2.clean_prices.height
        assert r1.returns.height == r2.returns.height
        assert r1.bhavcopy_paths
        assert r1.returns_paths
        assert "bhavcopy" in r1.duckdb_views or "returns" in r1.duckdb_views

        # BZ excluded; AAA retained with liquidation rows
        assert "JUNK" not in r1.clean_prices["symbol"].to_list()
        aaa = r1.clean_prices.filter(pl.col("symbol") == "AAA")
        assert aaa["is_liquidated"].any()

    def test_fundamentals_asof_no_lookahead(self) -> None:
        rets = pl.DataFrame(
            {
                "trade_date": [date(2020, 5, 1), date(2020, 8, 1)],
                "symbol": ["AAA", "AAA"],
                "ret": [0.01, 0.02],
            }
        )
        fund = pl.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "as_of_date": [date(2020, 3, 31), date(2020, 6, 30)],
                "available_date": [date(2020, 6, 30), date(2020, 9, 30)],
                "market_cap": [1.0e9, 2.0e9],
                "book_to_market": [0.5, 0.4],
            }
        )
        joined = join_fundamentals_asof(rets, fund)
        # May 1 is before first available_date -> null fundamentals
        may = joined.filter(pl.col("trade_date") == date(2020, 5, 1))
        assert may["market_cap"].null_count() == 1
        # Aug 1 can see Mar filing (available Jun 30), not Jun filing
        aug = joined.filter(pl.col("trade_date") == date(2020, 8, 1))
        assert aug["market_cap"][0] == pytest.approx(1.0e9)
