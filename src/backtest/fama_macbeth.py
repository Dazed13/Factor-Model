"""Fama–MacBeth regressions with Newey–West (HAC) standard errors.

Two modes
---------
1. **Characteristic FM** (primary, matches PLAN equation): each date $t$,
   cross-sectionally regress excess returns on characteristics (Size, Value,
   Momentum, ILLIQ). Report $\\bar{\\gamma}_k$ with Newey–West SEs / t-stats.

2. **Classical two-pass**: Pass 1 time-series betas vs factor returns; Pass 2
   cross-section of average returns (or period returns) on betas, then NW on
   the $\\lambda_t$ series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FMResult:
    """Aggregated Fama–MacBeth premium estimates."""

    lambdas: pl.DataFrame
    """Columns: name, mean, nw_se, t_stat, p_value, n_periods."""

    gamma_path: pl.DataFrame
    """Period-by-period γ / λ estimates (date + one column per regressor)."""

    factor_names: list[str] = field(default_factory=list)
    nw_lags: int = 0


def newey_west_mean_se(
    series: npt.NDArray[np.floating],
    *,
    lags: int | None = None,
) -> tuple[float, float, float, float]:
    """Mean of a time series with Newey–West HAC standard error.

    Returns ``(mean, nw_se, t_stat, p_value)`` from OLS on a constant with
    ``cov_type='HAC'``.
    """
    y = np.asarray(series, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    if lags is None:
        # Rule-of-thumb: floor(4 * (T/100)^(2/9))
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, lags)

    x = np.ones((n, 1))
    model = sm.OLS(y, x)
    if lags == 0:
        fit = model.fit()
    else:
        fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    mean = float(fit.params[0])
    se = float(fit.bse[0])
    t_stat = float(fit.tvalues[0]) if se > 0 else float("nan")
    p_value = float(fit.pvalues[0]) if np.isfinite(t_stat) else float("nan")
    return mean, se, t_stat, p_value


def _cross_section_ols(
    y: npt.NDArray[np.floating],
    x: npt.NDArray[np.floating],
    *,
    add_const: bool = True,
) -> npt.NDArray[np.floating] | None:
    """OLS coefficients for one cross-section; ``None`` if under-determined."""
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y_m = y[mask]
    x_m = x[mask]
    if y_m.size < x_m.shape[1] + (1 if add_const else 0) + 1:
        return None
    if add_const:
        x_m = sm.add_constant(x_m, has_constant="add")
    try:
        beta = sm.OLS(y_m, x_m).fit().params
    except Exception as exc:  # noqa: BLE001
        logger.debug("CS OLS failed: %s", exc)
        return None
    return np.asarray(beta, dtype=np.float64)


def fama_macbeth_characteristics(
    panel: pl.DataFrame,
    characteristic_cols: Sequence[str],
    *,
    date_col: str = "trade_date",
    ret_col: str = "excess_ret",
    add_intercept: bool = True,
    nw_lags: int | None = None,
    min_obs: int = 20,
) -> FMResult:
    """Characteristic-based FM: $R_{i,t}^e$ on chars within each date."""
    missing = [c for c in characteristic_cols if c not in panel.columns]
    if missing:
        raise KeyError(f"Missing characteristic columns: {missing}")
    if ret_col not in panel.columns:
        # Fall back to raw returns if excess not present
        if "ret" in panel.columns:
            ret_col = "ret"
            logger.warning("excess_ret missing — using ret for FM (document Rf coverage)")
        else:
            raise KeyError(f"Missing return column: {ret_col}")

    names = (["intercept"] if add_intercept else []) + list(characteristic_cols)
    rows: list[dict] = []

    for dt, group in panel.group_by(date_col, maintain_order=True):
        d = dt[0] if isinstance(dt, tuple) else dt
        g = group.drop_nulls(subset=[ret_col, *characteristic_cols])
        if g.height < min_obs:
            continue
        y = g.get_column(ret_col).to_numpy().astype(np.float64)
        x = g.select(characteristic_cols).to_numpy().astype(np.float64)
        beta = _cross_section_ols(y, x, add_const=add_intercept)
        if beta is None or beta.size != len(names):
            continue
        row: dict = {date_col: d}
        for name, val in zip(names, beta):
            row[name] = float(val)
        rows.append(row)

    if not rows:
        empty = pl.DataFrame(
            schema={
                "name": pl.Utf8,
                "mean": pl.Float64,
                "nw_se": pl.Float64,
                "t_stat": pl.Float64,
                "p_value": pl.Float64,
                "n_periods": pl.Float64,
            }
        )
        return FMResult(
            lambdas=empty,
            gamma_path=pl.DataFrame(),
            factor_names=list(characteristic_cols),
            nw_lags=0,
        )

    gamma_path = pl.DataFrame(rows).sort(date_col)
    summary_rows: list[dict] = []
    used_lags = 0
    for name in names:
        series = gamma_path.get_column(name).to_numpy().astype(np.float64)
        mean, se, t_stat, p_value = newey_west_mean_se(series, lags=nw_lags)
        if nw_lags is None:
            used_lags = int(np.floor(4 * (np.isfinite(series).sum() / 100.0) ** (2.0 / 9.0)))
        else:
            used_lags = nw_lags
        summary_rows.append(
            {
                "name": name,
                "mean": mean,
                "nw_se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_periods": float(np.isfinite(series).sum()),
            }
        )

    return FMResult(
        lambdas=pl.DataFrame(summary_rows),
        gamma_path=gamma_path,
        factor_names=list(characteristic_cols),
        nw_lags=used_lags,
    )


def time_series_betas(
    returns: pl.DataFrame,
    factors: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    ret_col: str = "excess_ret",
    factor_cols: Sequence[str] | None = None,
    min_obs: int = 60,
) -> pl.DataFrame:
    """Pass 1: per-symbol OLS betas of asset excess returns on factor returns."""
    if ret_col not in returns.columns:
        ret_col = "ret" if "ret" in returns.columns else ret_col
    fac = factors
    if factor_cols is None:
        factor_cols = [c for c in fac.columns if c != date_col]
    fac = fac.select([date_col, *factor_cols])

    merged = returns.select([date_col, symbol_col, ret_col]).join(
        fac, on=date_col, how="inner"
    )
    rows: list[dict] = []
    for sym, group in merged.group_by(symbol_col, maintain_order=True):
        s = sym[0] if isinstance(sym, tuple) else sym
        g = group.drop_nulls()
        if g.height < min_obs:
            continue
        y = g.get_column(ret_col).to_numpy().astype(np.float64)
        x = g.select(factor_cols).to_numpy().astype(np.float64)
        beta = _cross_section_ols(y, x, add_const=True)
        if beta is None:
            continue
        row: dict = {symbol_col: s, "alpha": float(beta[0])}
        for i, name in enumerate(factor_cols):
            row[f"beta_{name}"] = float(beta[i + 1])
        rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def fama_macbeth_two_pass(
    returns: pl.DataFrame,
    factors: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    ret_col: str = "excess_ret",
    factor_cols: Sequence[str] | None = None,
    nw_lags: int | None = None,
    min_ts_obs: int = 60,
    min_cs_obs: int = 20,
) -> FMResult:
    """Classical FM: Pass-1 betas, Pass-2 cross-section of period returns on betas."""
    if factor_cols is None:
        factor_cols = [c for c in factors.columns if c != date_col]

    betas = time_series_betas(
        returns,
        factors,
        date_col=date_col,
        symbol_col=symbol_col,
        ret_col=ret_col,
        factor_cols=factor_cols,
        min_obs=min_ts_obs,
    )
    if betas.is_empty():
        return FMResult(
            lambdas=pl.DataFrame(),
            gamma_path=pl.DataFrame(),
            factor_names=list(factor_cols),
            nw_lags=0,
        )

    beta_cols = [f"beta_{c}" for c in factor_cols]
    names = ["intercept", *factor_cols]

    if ret_col not in returns.columns:
        ret_col = "ret"

    panel = returns.select([date_col, symbol_col, ret_col]).join(
        betas, on=symbol_col, how="inner"
    )

    rows: list[dict] = []
    for dt, group in panel.group_by(date_col, maintain_order=True):
        d = dt[0] if isinstance(dt, tuple) else dt
        g = group.drop_nulls(subset=[ret_col, *beta_cols])
        if g.height < min_cs_obs:
            continue
        y = g.get_column(ret_col).to_numpy().astype(np.float64)
        x = g.select(beta_cols).to_numpy().astype(np.float64)
        lam = _cross_section_ols(y, x, add_const=True)
        if lam is None or lam.size != len(names):
            continue
        row: dict = {date_col: d}
        for name, val in zip(names, lam):
            row[name] = float(val)
        rows.append(row)

    gamma_path = pl.DataFrame(rows).sort(date_col) if rows else pl.DataFrame()
    summary_rows: list[dict] = []
    used_lags = nw_lags or 0
    for name in names:
        if gamma_path.is_empty() or name not in gamma_path.columns:
            continue
        series = gamma_path.get_column(name).to_numpy().astype(np.float64)
        mean, se, t_stat, p_value = newey_west_mean_se(series, lags=nw_lags)
        if nw_lags is None:
            used_lags = int(np.floor(4 * (np.isfinite(series).sum() / 100.0) ** (2.0 / 9.0)))
        summary_rows.append(
            {
                "name": name,
                "mean": mean,
                "nw_se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_periods": float(np.isfinite(series).sum()),
            }
        )

    return FMResult(
        lambdas=pl.DataFrame(summary_rows),
        gamma_path=gamma_path,
        factor_names=list(factor_cols),
        nw_lags=used_lags,
    )


def ols_mean_se_for_contrast(
    series: npt.NDArray[np.floating],
) -> tuple[float, float]:
    """Plain OLS SE of the mean (for tests contrasting vs Newey–West)."""
    y = np.asarray(series, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n))
    return mean, se
