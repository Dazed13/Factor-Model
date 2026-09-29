"""Unit tests for Amihud ILLIQ: zero-volume safety and lookahead lag."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.factors.amihud import (
    attach_daily_illiq,
    attach_illiq_trade_signal,
    daily_amihud_illiq,
)


def _panel(n_days: int = 40) -> pl.DataFrame:
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    # skip weekends roughly by still using consecutive calendar days for unit tests
    ret = np.linspace(-0.02, 0.02, n_days)
    vol = np.full(n_days, 1.0e7)
    vol[5] = 0.0  # zero-volume day
    vol[6] = np.nan
    return pl.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["AAA"] * n_days,
            "ret": ret,
            "tottrdval": vol,
        }
    )


class TestDailyAmihud:
    def test_zero_volume_returns_nan_no_exception(self) -> None:
        out = daily_amihud_illiq(0.05, 0.0)
        assert np.isnan(out)

    def test_zero_volume_option_zero(self) -> None:
        out = daily_amihud_illiq(0.05, 0.0, zero_volume=0.0)
        assert out == pytest.approx(0.0)

    def test_positive_volume(self) -> None:
        assert daily_amihud_illiq(0.02, 1_000_000.0) == pytest.approx(0.02 / 1_000_000.0)

    def test_vectorized_no_zerodivision(self) -> None:
        r = np.array([0.01, -0.02, 0.03])
        v = np.array([1e6, 0.0, 2e6])
        out = daily_amihud_illiq(r, v)
        assert isinstance(out, np.ndarray)
        assert np.isnan(out[1])
        assert out[0] == pytest.approx(1e-8)


class TestPolarsAmihud:
    def test_attach_daily_null_on_zero_volume(self) -> None:
        df = attach_daily_illiq(_panel())
        assert df.filter(pl.col("tottrdval") <= 0)["illiq"].null_count() == 1
        # np.nan volumes are non-finite — must also yield null ILLIQ
        assert df.filter(~pl.col("tottrdval").is_finite())["illiq"].null_count() >= 1

    def test_trade_signal_is_shifted(self) -> None:
        df = attach_illiq_trade_signal(_panel(n_days=40), window=5)
        # Signal at t equals rolling mean at t-1
        row_t = df.row(20, named=True)
        row_tm1 = df.row(19, named=True)
        if row_tm1["illiq_roll"] is not None and row_t["illiq_signal"] is not None:
            assert row_t["illiq_signal"] == pytest.approx(row_tm1["illiq_roll"])
        # First row signal must be null (no t-1)
        assert df["illiq_signal"][0] is None

    def test_signal_does_not_use_same_day_illiq(self) -> None:
        """Perturb day-t ILLIQ; lagged signal at t must be unchanged."""
        base = attach_illiq_trade_signal(_panel(n_days=40), window=5)
        # Manually inflate ret on day 20 → illiq changes that day
        tweaked = _panel(n_days=40).with_columns(
            pl.when(pl.col("trade_date") == date(2020, 1, 2) + timedelta(days=20))
            .then(pl.lit(0.50))
            .otherwise(pl.col("ret"))
            .alias("ret")
        )
        alt = attach_illiq_trade_signal(tweaked, window=5)
        # Signal on day 20 uses data through day 19 only
        assert base["illiq_signal"][20] == pytest.approx(alt["illiq_signal"][20])
        # But same-day rolling mean may differ
        assert base["illiq_roll"][20] != pytest.approx(alt["illiq_roll"][20])
