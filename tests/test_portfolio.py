"""Tests for dollar-neutral portfolios, sorts, turnover, and backtest engine."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.backtest.engine import BacktestConfig, propagate_weights, run_backtest, run_quantile_spread
from src.backtest.portfolio import (
    WEIGHT_SUM_TOL,
    assert_dollar_neutral,
    form_quantile_portfolio,
    weights_sum_by_date,
)
from src.backtest.sorts import assign_sorts, lag_signal
from src.backtest.turnover import average_turnover, compute_turnover


def _panel(n_symbols: int = 20, n_days: int = 60) -> pl.DataFrame:
    start = date(2020, 1, 2)
    rows: list[dict] = []
    rng = np.random.default_rng(0)
    for d in range(n_days):
        dt = start + timedelta(days=d)
        for i in range(n_symbols):
            signal = float(i) + 0.01 * d  # cross-sectional ranking stable
            rows.append(
                {
                    "trade_date": dt,
                    "symbol": f"S{i:02d}",
                    "ret": float(rng.normal(0.001 * (i - n_symbols / 2) / n_symbols, 0.01)),
                    "illiq_signal": signal,
                    "market_cap": 1e9 * (i + 1),
                }
            )
    return pl.DataFrame(rows)


class TestSorts:
    def test_quintile_range(self) -> None:
        df = _panel(n_symbols=25, n_days=3)
        out = assign_sorts(df, "illiq_signal", kind="quintile")
        qs = out.filter(pl.col("trade_date") == date(2020, 1, 2))["quantile"].drop_nulls()
        assert set(qs.to_list()) == {1, 2, 3, 4, 5}

    def test_lag_signal(self) -> None:
        df = pl.DataFrame(
            {
                "trade_date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)],
                "symbol": ["A", "A", "A"],
                "raw": [1.0, 2.0, 3.0],
            }
        )
        out = lag_signal(df, "raw", periods=1)
        assert out["raw_lag1"].to_list()[0] is None
        assert out["raw_lag1"].to_list()[1] == pytest.approx(1.0)


class TestPortfolio:
    def test_dollar_neutral_equal_weight(self) -> None:
        df = _panel(n_symbols=20, n_days=5)
        # Single formation date
        day = df.filter(pl.col("trade_date") == date(2020, 1, 2))
        w = form_quantile_portfolio(
            day, "illiq_signal", kind="quintile", long_high=True, weighting="equal"
        )
        assert_dollar_neutral(w)
        sums = weights_sum_by_date(w)
        assert abs(sums["w_sum"][0]) < WEIGHT_SUM_TOL
        assert sums["gross"][0] == pytest.approx(1.0)

        longs = w.filter(pl.col("weight") > 0)
        shorts = w.filter(pl.col("weight") < 0)
        assert longs.height > 0 and shorts.height > 0
        assert (longs["weight"] > 0).all()
        assert (shorts["weight"] < 0).all()
        assert abs(float(longs["weight"].sum()) - 0.5) < 1e-9
        assert abs(float(shorts["weight"].sum()) + 0.5) < 1e-9

    def test_dollar_neutral_value_weight(self) -> None:
        day = _panel(n_symbols=20, n_days=1)
        w = form_quantile_portfolio(
            day,
            "illiq_signal",
            kind="quintile",
            long_high=True,
            weighting="value",
            weight_col="market_cap",
        )
        assert_dollar_neutral(w)
        assert abs(float(w["weight"].sum())) < WEIGHT_SUM_TOL

    def test_long_low_for_size_style(self) -> None:
        day = _panel(n_symbols=20, n_days=1)
        w = form_quantile_portfolio(
            day, "illiq_signal", kind="quintile", long_high=False, weighting="equal"
        )
        # Long quantile 1 (low signal)
        assert w.filter(pl.col("is_long"))["quantile"].unique().to_list() == [1]
        assert w.filter(pl.col("is_short"))["quantile"].unique().to_list() == [5]


class TestTurnover:
    def test_turnover_positive_on_change(self) -> None:
        w = pl.DataFrame(
            {
                "trade_date": [date(2020, 1, 31), date(2020, 1, 31), date(2020, 2, 28), date(2020, 2, 28)],
                "symbol": ["A", "B", "A", "C"],
                "weight": [0.5, -0.5, 0.5, -0.5],
            }
        )
        to = compute_turnover(w)
        assert to.height == 2
        assert to["turnover_two_way"][0] == pytest.approx(0.0)
        assert to["turnover_two_way"][1] > 0
        avg = average_turnover(to)
        assert avg["turnover_two_way"] > 0


class TestEngine:
    def test_run_backtest_end_to_end(self) -> None:
        panel = _panel(n_symbols=20, n_days=90)
        cfg = BacktestConfig(
            signal_col="illiq_signal",
            kind="quintile",
            long_high=True,
            weighting="equal",
            frequency="month_end",
            ensure_lag=False,  # already a lagged-style signal in synthetic data
        )
        result = run_backtest(panel, cfg)
        assert result.weights.height > 0
        assert_dollar_neutral(result.weights)
        assert result.returns.height > 0
        assert "net_ret" in result.returns.columns
        assert "wealth" in result.returns.columns
        assert result.metrics["n_rebals"] >= 2

    def test_propagate_shifts_weights(self) -> None:
        formation = pl.DataFrame(
            {
                "trade_date": [date(2020, 1, 31), date(2020, 1, 31)],
                "symbol": ["A", "B"],
                "weight": [0.5, -0.5],
                "quantile": [5, 1],
                "is_long": [True, False],
                "is_short": [False, True],
            }
        )
        calendar = [date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 4)]
        daily = propagate_weights(formation, calendar)
        # No weight on formation day after shift
        assert daily.filter(pl.col("trade_date") == date(2020, 1, 31)).height == 0
        # Weights appear from next day
        held = daily.filter(pl.col("trade_date") == date(2020, 2, 3))
        assert held.height == 2
        assert abs(float(held["weight"].sum())) < WEIGHT_SUM_TOL

    def test_quantile_spread_shape(self) -> None:
        panel = _panel(n_symbols=20, n_days=5)
        spread = run_quantile_spread(panel, "illiq_signal", kind="quintile")
        assert set(spread["quantile"].unique().to_list()) == {1, 2, 3, 4, 5}
