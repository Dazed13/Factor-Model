"""Unit tests for SMB / HML / WML / ILLIQ factor construction."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.factors.combine import build_characteristic_panel, build_factor_returns, build_factors
from src.factors.momentum import attach_momentum_characteristic, construct_wml
from src.factors.size import construct_smb
from src.factors.value import construct_hml
from src.factors.winsorize import winsorize_cross_section, zscore_cross_section


def _cross_section(n_symbols: int = 20, n_days: int = 30) -> pl.DataFrame:
    """Synthetic panel with ME, BTM, volume, and returns."""
    start = date(2020, 1, 2)
    rows: list[dict] = []
    rng = np.random.default_rng(42)
    for d in range(n_days):
        dt = start + timedelta(days=d)
        for i in range(n_symbols):
            me = 1e9 * (i + 1)
            btm = 0.2 + 0.05 * i
            ret = float(rng.normal(0.0005 - 0.0001 * i, 0.01))
            rows.append(
                {
                    "trade_date": dt,
                    "symbol": f"S{i:02d}",
                    "ret": ret,
                    "tottrdval": 1e7 * (n_symbols - i),
                    "market_cap": me,
                    "book_to_market": btm,
                }
            )
    return pl.DataFrame(rows)


class TestWinsorize:
    def test_percentile_clips_outliers(self) -> None:
        df = pl.DataFrame(
            {
                "trade_date": [date(2020, 1, 2)] * 100,
                "x": list(range(100)),
            }
        )
        out = winsorize_cross_section(df, "x", lower=0.01, upper=0.99)
        assert out["x"].min() >= 1
        assert out["x"].max() <= 98

    def test_zscore_mean_near_zero(self) -> None:
        df = pl.DataFrame(
            {
                "trade_date": [date(2020, 1, 2)] * 50,
                "x": list(range(50)),
            }
        )
        out = zscore_cross_section(df, "x")
        assert abs(out["x_z"].mean()) < 1e-8


class TestSMBHMLWML:
    def test_smb_negative_when_big_outperforms(self) -> None:
        # Big names (high ME) get +ret, small get -ret → SMB < 0
        df = _cross_section(n_symbols=10, n_days=5).with_columns(
            pl.when(pl.col("market_cap") > 5e9)
            .then(pl.lit(0.02))
            .otherwise(pl.lit(-0.02))
            .alias("ret")
        )
        df = df.with_columns(pl.col("market_cap").alias("me"))
        smb = construct_smb(df)
        assert smb.height == 5
        assert (smb["SMB"] < 0).all()

    def test_hml_positive_when_high_btm_outperforms(self) -> None:
        df = _cross_section(n_symbols=10, n_days=5).with_columns(
            pl.when(pl.col("book_to_market") >= 0.55)
            .then(pl.lit(0.03))
            .otherwise(pl.lit(-0.01))
            .alias("ret"),
            pl.col("market_cap").alias("me"),
            pl.col("book_to_market").alias("btm"),
        )
        hml = construct_hml(df)
        assert hml.height == 5
        assert (hml["HML"] > 0).all()

    def test_momentum_characteristic_shape(self) -> None:
        df = _cross_section(n_symbols=5, n_days=280)
        out = attach_momentum_characteristic(df, lookback=60, skip=21)
        assert "mom_12_1" in out.columns
        # Early rows should be null due to lookback/skip
        assert out.filter(pl.col("symbol") == "S00")["mom_12_1"].null_count() > 0
        # Later rows populated
        assert out.filter(pl.col("symbol") == "S00")["mom_12_1"].drop_nulls().len() > 0

    def test_wml_runs(self) -> None:
        df = _cross_section(n_symbols=12, n_days=100)
        df = attach_momentum_characteristic(df, lookback=40, skip=10)
        wml = construct_wml(df)
        assert "WML" in wml.columns
        assert wml["WML"].drop_nulls().len() > 0


class TestCombine:
    def test_build_factors_end_to_end(self, tmp_path) -> None:
        df = _cross_section(n_symbols=15, n_days=80)
        result = build_factors(
            df, amihud_window=5, winsorize=True, dest_dir=tmp_path, persist=True
        )
        assert result.characteristics.height == df.height
        assert "illiq_signal" in result.characteristics.columns
        assert "mom_12_1" in result.characteristics.columns
        for col in ("SMB", "HML", "WML", "ILLIQ"):
            assert col in result.factor_returns.columns
        assert result.characteristic_paths
        assert result.factor_paths

    def test_factor_returns_aligned_on_date(self) -> None:
        df = _cross_section(n_symbols=12, n_days=50)
        chars = build_characteristic_panel(df, amihud_window=5, winsorize=False)
        factors = build_factor_returns(chars)
        assert factors["trade_date"].is_sorted()
        assert factors.height <= 50
