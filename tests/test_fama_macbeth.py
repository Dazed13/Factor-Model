"""Fama–MacBeth + Newey–West unit tests (synthetic premium recovery)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.backtest.fama_macbeth import (
    fama_macbeth_characteristics,
    fama_macbeth_two_pass,
    newey_west_mean_se,
    ols_mean_se_for_contrast,
    time_series_betas,
)
from src.backtest.report import export_fm_report, to_markdown_table


def _char_panel(
    n_dates: int = 80,
    n_symbols: int = 40,
    true_premia: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Panel where excess_ret = γ0 + γ·chars + noise (chars ~ N(0,1))."""
    true_premia = true_premia or {"SMB": 0.002, "HML": 0.0015, "ILLIQ": 0.001}
    rng = np.random.default_rng(123)
    start = date(2018, 1, 2)
    rows: list[dict] = []
    for t in range(n_dates):
        dt = start + timedelta(days=t)
        for i in range(n_symbols):
            smb = float(rng.normal())
            hml = float(rng.normal())
            illiq = float(rng.normal())
            noise = float(rng.normal(0, 0.01))
            ret = (
                0.0002
                + true_premia["SMB"] * smb
                + true_premia["HML"] * hml
                + true_premia["ILLIQ"] * illiq
                + noise
            )
            rows.append(
                {
                    "trade_date": dt,
                    "symbol": f"S{i:02d}",
                    "excess_ret": ret,
                    "SMB": smb,
                    "HML": hml,
                    "ILLIQ": illiq,
                }
            )
    return pl.DataFrame(rows)


class TestNeweyWest:
    def test_nw_returns_finite(self) -> None:
        rng = np.random.default_rng(0)
        # AR(1) series so HAC differs from OLS
        e = rng.normal(size=200)
        y = np.zeros(200)
        for i in range(1, 200):
            y[i] = 0.5 * y[i - 1] + e[i]
        y = y + 0.01
        mean, se, t_stat, p_value = newey_west_mean_se(y, lags=5)
        assert np.isfinite(mean) and np.isfinite(se) and np.isfinite(t_stat)
        assert se > 0
        ols_mean, ols_se = ols_mean_se_for_contrast(y)
        # HAC SE should generally differ from iid OLS SE for persistent series
        assert se != pytest.approx(ols_se, rel=1e-12)


class TestCharacteristicFM:
    def test_recovers_known_premia(self) -> None:
        true = {"SMB": 0.002, "HML": 0.0015, "ILLIQ": 0.001}
        panel = _char_panel(n_dates=100, n_symbols=50, true_premia=true)
        result = fama_macbeth_characteristics(
            panel,
            ["SMB", "HML", "ILLIQ"],
            ret_col="excess_ret",
            nw_lags=4,
            min_obs=20,
        )
        assert not result.lambdas.is_empty()
        assert result.nw_lags == 4
        by_name = {r["name"]: r for r in result.lambdas.to_dicts()}
        for k, v in true.items():
            assert by_name[k]["mean"] == pytest.approx(v, rel=0.35, abs=0.0008)
            assert np.isfinite(by_name[k]["t_stat"])
            assert by_name[k]["nw_se"] > 0

    def test_gamma_path_has_dates(self) -> None:
        panel = _char_panel(n_dates=40, n_symbols=30)
        result = fama_macbeth_characteristics(panel, ["SMB", "HML", "ILLIQ"])
        assert result.gamma_path.height >= 30
        assert "trade_date" in result.gamma_path.columns


class TestTwoPass:
    def test_time_series_betas_shape(self) -> None:
        rng = np.random.default_rng(1)
        start = date(2019, 1, 2)
        n_t, n_i = 120, 15
        dates = [start + timedelta(days=t) for t in range(n_t)]
        # Single market factor
        mkt = rng.normal(0.0005, 0.01, size=n_t)
        factors = pl.DataFrame({"trade_date": dates, "MKT": mkt})
        rows = []
        true_beta = {f"S{i:02d}": 0.5 + 0.1 * i for i in range(n_i)}
        for t, dt in enumerate(dates):
            for i in range(n_i):
                sym = f"S{i:02d}"
                rows.append(
                    {
                        "trade_date": dt,
                        "symbol": sym,
                        "excess_ret": true_beta[sym] * mkt[t] + rng.normal(0, 0.005),
                    }
                )
        rets = pl.DataFrame(rows)
        betas = time_series_betas(rets, factors, factor_cols=["MKT"], min_obs=60)
        assert betas.height == n_i
        assert "beta_MKT" in betas.columns
        # Recover betas roughly
        for row in betas.to_dicts():
            assert row["beta_MKT"] == pytest.approx(true_beta[row["symbol"]], rel=0.25)

    def test_two_pass_runs(self) -> None:
        rng = np.random.default_rng(2)
        start = date(2019, 1, 2)
        n_t, n_i = 100, 25
        dates = [start + timedelta(days=t) for t in range(n_t)]
        mkt = rng.normal(0.001, 0.01, size=n_t)
        factors = pl.DataFrame({"trade_date": dates, "MKT": mkt})
        rows = []
        for t, dt in enumerate(dates):
            for i in range(n_i):
                beta = 0.8 + 0.02 * i
                rows.append(
                    {
                        "trade_date": dt,
                        "symbol": f"S{i:02d}",
                        "excess_ret": 0.0005 + beta * mkt[t] + rng.normal(0, 0.008),
                    }
                )
        rets = pl.DataFrame(rows)
        result = fama_macbeth_two_pass(
            rets, factors, factor_cols=["MKT"], min_ts_obs=50, min_cs_obs=15, nw_lags=3
        )
        assert not result.lambdas.is_empty()
        assert "MKT" in result.lambdas["name"].to_list()


class TestReport:
    def test_markdown_and_export(self, tmp_path) -> None:
        panel = _char_panel(n_dates=40, n_symbols=25)
        result = fama_macbeth_characteristics(panel, ["SMB", "HML", "ILLIQ"], nw_lags=2)
        paths = export_fm_report(result, dest_dir=tmp_path, stem="test_fm")
        assert paths["lambdas"].exists()
        assert paths["markdown"].exists()
        md = to_markdown_table(result.lambdas)
        assert "t_stat" in md
        assert "|" in md
