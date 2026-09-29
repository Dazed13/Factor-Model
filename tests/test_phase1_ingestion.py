"""Unit tests for Phase-1 symbol, bhavcopy, universe, RF, and CA helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.data.bhavcopy import (
    ALLOWED_SERIES,
    detect_format,
    load_bhavcopy_file,
    parse_old_bhavcopy,
    parse_udiff_bhavcopy,
    write_bhavcopy_parquet,
)
from src.data.corporate_actions import (
    adjusted_returns,
    adjustment_factor_from_prices,
    apply_adjustment_factor,
)
from src.data.fundamentals import (
    lag_available_date,
    load_fundamentals_csv,
    point_in_time_fundamentals,
)
from src.data.risk_free import (
    annualized_yield_to_daily,
    attach_excess_returns,
    build_risk_free_frame,
    load_risk_free_csv,
)
from src.data.symbols import SymbolMap, strip_exchange_suffix, to_yfinance_symbols
from src.data.universe import point_in_time_membership


class TestSymbols:
    def test_strip_exchange_suffix(self) -> None:
        assert strip_exchange_suffix("reliance.ns") == "RELIANCE"
        assert strip_exchange_suffix("TCS") == "TCS"

    def test_yfinance_roundtrip(self) -> None:
        sm = SymbolMap()
        assert sm.to_yfinance("RELIANCE") == "RELIANCE.NS"
        assert sm.from_yfinance("RELIANCE.NS") == "RELIANCE"
        assert to_yfinance_symbols(["tcs", "infy.ns"]) == ["TCS.NS", "INFY.NS"]

    def test_normalize_frame(self) -> None:
        sm = SymbolMap()
        df = pl.DataFrame({"symbol": ["reliance.ns", "HDFC"]})
        out = sm.normalize_frame(df)
        assert out["symbol"].to_list() == ["RELIANCE", "HDFCBANK"]


class TestBhavcopy:
    def test_detect_formats(self) -> None:
        assert detect_format(["SYMBOL", "SERIES", "OPEN", "CLOSE"]) == "old"
        assert detect_format(["TradDt", "TckrSymb", "SctySrs", "OpnPric"]) == "udiff"

    def test_old_parser_filters_series(self, old_bhav_csv_bytes: bytes) -> None:
        df = parse_old_bhavcopy(old_bhav_csv_bytes, trade_date=date(2020, 1, 2))
        assert set(df["series"].unique().to_list()) <= ALLOWED_SERIES
        assert "ILLIQUID" not in df["symbol"].to_list()
        assert "DEBTCO" not in df["symbol"].to_list()
        assert set(df["symbol"].to_list()) == {"RELIANCE", "TCS", "TRADETT"}
        assert df.filter(pl.col("symbol") == "RELIANCE")["close"][0] == pytest.approx(2520.0)

    def test_udiff_parser_filters_series(self, udiff_bhav_csv_bytes: bytes) -> None:
        df = parse_udiff_bhavcopy(udiff_bhav_csv_bytes, trade_date=date(2024, 7, 8))
        assert set(df["series"].unique().to_list()) <= ALLOWED_SERIES
        assert "ILLIQUID" not in df["symbol"].to_list()
        assert df["trade_date"].dtype == pl.Date

    def test_zip_roundtrip(
        self, fixtures_dir: Path, old_bhav_zip_bytes: bytes  # noqa: ARG002
    ) -> None:
        old_path = fixtures_dir / "cm02JAN2020bhav.csv.zip"
        udiff_path = fixtures_dir / "BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip"
        old_df = load_bhavcopy_file(old_path)
        udiff_df = load_bhavcopy_file(udiff_path)
        assert old_df.height == 3
        assert udiff_df.height == 3
        assert "BZ" not in old_df["series"].to_list()

    def test_write_parquet_partition(self, old_bhav_csv_bytes: bytes, tmp_path: Path) -> None:
        df = parse_old_bhavcopy(old_bhav_csv_bytes, trade_date=date(2020, 1, 2))
        paths = write_bhavcopy_parquet(df, dest_dir=tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()
        reloaded = pl.read_parquet(paths[0])
        assert reloaded.height == df.height


class TestUniverse:
    def test_point_in_time_no_lookahead(self, sample_universe: pl.DataFrame) -> None:
        m = point_in_time_membership(sample_universe, date(2020, 6, 1))
        assert m.select(pl.col("as_of_date").unique()).item() == date(2020, 1, 1)
        assert set(m["symbol"].to_list()) == {"RELIANCE", "TCS"}

        m2 = point_in_time_membership(sample_universe, date(2021, 7, 1))
        assert m2.select(pl.col("as_of_date").unique()).item() == date(2021, 6, 1)
        assert m2["symbol"].to_list() == ["RELIANCE"]


class TestRiskFree:
    def test_yield_conversion_percent_and_decimal(self) -> None:
        assert annualized_yield_to_daily(0.073) == pytest.approx(0.073 / 365)
        assert annualized_yield_to_daily(7.3) == pytest.approx(0.073 / 365)

    def test_build_and_excess(self) -> None:
        rf = build_risk_free_frame(
            [date(2020, 1, 2), date(2020, 1, 3)],
            [6.5, 6.5],
            source="rbi_91d_tbill",
        )
        assert rf["source"].unique().to_list() == ["rbi_91d_tbill"]
        rets = pl.DataFrame(
            {
                "trade_date": [date(2020, 1, 2), date(2020, 1, 3)],
                "symbol": ["RELIANCE", "RELIANCE"],
                "ret": [0.01, -0.005],
            }
        )
        out = attach_excess_returns(rets, rf)
        assert "excess_ret" in out.columns
        assert out["excess_ret"][0] == pytest.approx(0.01 - 6.5 / 100 / 365)

    def test_rejects_us_source(self) -> None:
        rf = build_risk_free_frame([date(2020, 1, 2)], [0.05], source="rbi_91d_tbill")
        rf = rf.with_columns(pl.lit("sofr").alias("source"))
        rets = pl.DataFrame(
            {"trade_date": [date(2020, 1, 2)], "symbol": ["X"], "ret": [0.01]}
        )
        with pytest.raises(ValueError, match="US risk-free"):
            attach_excess_returns(rets, rf)

    def test_load_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "rf.csv"
        path.write_text("date,yield\n2024-01-05,6.75\n")
        df = load_risk_free_csv(path)
        assert df.height == 1
        assert df["daily_rf"][0] == pytest.approx(0.0675 / 365)


class TestCorporateActions:
    def test_adjustment_factor(self) -> None:
        close = np.array([100.0, 50.0, 0.0, np.nan])
        adj = np.array([100.0, 100.0, 10.0, 1.0])
        factor = adjustment_factor_from_prices(close, adj)
        assert factor[0] == pytest.approx(1.0)
        assert factor[1] == pytest.approx(2.0)
        assert np.isnan(factor[2])
        assert np.isnan(factor[3])

    def test_apply_and_returns_use_adj_close(self) -> None:
        df = pl.DataFrame(
            {
                "trade_date": [date(2020, 1, 2), date(2020, 1, 3)],
                "symbol": ["RELIANCE", "RELIANCE"],
                "close": [100.0, 50.0],
                "adj_factor": [1.0, 2.0],
            }
        )
        adj = apply_adjustment_factor(df, price_cols=["close"])
        assert "adj_close" in adj.columns
        assert adj["adj_close"].to_list() == [100.0, 100.0]
        rets = adjusted_returns(adj)
        assert rets["ret"][1] == pytest.approx(0.0)


class TestFundamentals:
    def test_lag_months(self) -> None:
        assert lag_available_date(date(2020, 1, 31), 3) == date(2020, 4, 30)

    def test_pit_fundamentals(self, tmp_path: Path) -> None:
        path = tmp_path / "f.csv"
        path.write_text(
            "as_of_date,symbol,market_cap,book_to_market\n"
            "2020-03-31,RELIANCE,1e12,0.4\n"
            "2020-06-30,RELIANCE,1.1e12,0.35\n"
        )
        df = load_fundamentals_csv(path, lag_months=3)
        assert "available_date" in df.columns
        # 2020-03-31 + 3m -> 2020-06-30
        pit = point_in_time_fundamentals(df, date(2020, 7, 15))
        assert pit.height == 1
        assert pit["as_of_date"][0] == date(2020, 3, 31)

        # Before first available_date -> empty
        early = point_in_time_fundamentals(df, date(2020, 5, 1))
        assert early.height == 0
