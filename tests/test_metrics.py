"""Unit tests for Sharpe, drawdown, and related performance metrics."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.backtest.metrics import (
    TRADING_DAYS,
    annualized_sharpe,
    cumulative_wealth,
    hit_rate,
    max_drawdown,
    quantile_metrics,
    summarize_returns,
)


class TestSharpe:
    def test_annualization_factor(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, size=252)
        sr = annualized_sharpe(r)
        # Manual check
        expected = np.sqrt(TRADING_DAYS) * np.mean(r) / np.std(r, ddof=1)
        assert sr == pytest.approx(expected)

    def test_constant_returns_nan_vol(self) -> None:
        assert np.isnan(annualized_sharpe(np.ones(10) * 0.01))


class TestDrawdown:
    def test_max_drawdown_bounds(self) -> None:
        wealth = np.array([1.0, 1.2, 0.9, 1.1, 0.8])
        mdd = max_drawdown(wealth)
        assert mdd == pytest.approx(0.8 / 1.2 - 1.0)
        assert -1.0 <= mdd <= 0.0

    def test_no_drawdown(self) -> None:
        wealth = cumulative_wealth(np.array([0.01, 0.01, 0.01]))
        assert max_drawdown(wealth) == pytest.approx(0.0)


class TestSummaries:
    def test_hit_rate(self) -> None:
        assert hit_rate(np.array([0.1, -0.1, 0.2, 0.0])) == pytest.approx(0.5)

    def test_summarize_returns_keys(self) -> None:
        stats = summarize_returns(np.array([0.01, -0.005, 0.002, 0.003]))
        for key in ("sharpe", "max_drawdown", "hit_rate", "ann_vol", "n_obs"):
            assert key in stats

    def test_quantile_metrics_shape(self) -> None:
        start = date(2020, 1, 2)
        rng = np.random.default_rng(0)
        rows = []
        for d in range(60):
            for q in range(1, 6):
                rows.append(
                    {
                        "trade_date": start + timedelta(days=d),
                        "quantile": q,
                        "q_ret": 0.001 * q + float(rng.normal(0, 0.01)),
                    }
                )
        df = pl.DataFrame(rows)
        out = quantile_metrics(df)
        assert out.height == 5
        # Higher quantile mean return → higher Sharpe in expectation
        assert out["ann_mean"][-1] > out["ann_mean"][0]
        assert np.isfinite(out["sharpe"].to_numpy()).all()
