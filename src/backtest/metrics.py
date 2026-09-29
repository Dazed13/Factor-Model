"""Performance metrics for factor portfolios and quantile sleeves.

Annualized Sharpe uses $\\sqrt{252}$ for daily returns per quant-math rules.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import numpy.typing as npt
import polars as pl

from src.backtest.turnover import average_turnover, compute_turnover

TRADING_DAYS: float = 252.0


def annualized_sharpe(
    returns: npt.NDArray[np.floating] | pl.Series | list[float],
    *,
    periods_per_year: float = TRADING_DAYS,
    risk_free: float = 0.0,
) -> float:
    """Sharpe ratio: $\\sqrt{N}\\,\\bar{r}/\\hat{\\sigma}$ on demeaned excess returns."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - risk_free
    mu = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=1))
    if not np.isfinite(sigma) or sigma < 1e-12:
        return float("nan")
    return np.sqrt(periods_per_year) * mu / sigma


def cumulative_wealth(
    returns: npt.NDArray[np.floating] | pl.Series,
    *,
    start: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Wealth path $W_t = W_0 \\prod (1+r)$."""
    r = np.asarray(returns, dtype=np.float64)
    r = np.where(np.isfinite(r), r, 0.0)
    return start * np.cumprod(1.0 + r)


def max_drawdown(wealth: npt.NDArray[np.floating] | pl.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.25 = -25%)."""
    w = np.asarray(wealth, dtype=np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(w)
    dd = w / peak - 1.0
    return float(np.min(dd))


def hit_rate(returns: npt.NDArray[np.floating] | pl.Series) -> float:
    """Fraction of periods with strictly positive return."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > 0))


def summarize_returns(
    returns: pl.DataFrame | npt.NDArray[np.floating] | pl.Series,
    *,
    ret_col: str = "net_ret",
    periods_per_year: float = TRADING_DAYS,
) -> dict[str, float]:
    """Sharpe, drawdown, hit rate, mean/vol for a return series or frame."""
    if isinstance(returns, pl.DataFrame):
        if ret_col not in returns.columns:
            raise KeyError(f"Missing return column: {ret_col}")
        r = returns.get_column(ret_col).to_numpy().astype(np.float64)
    else:
        r = np.asarray(returns, dtype=np.float64)

    r_finite = r[np.isfinite(r)]
    wealth = cumulative_wealth(r_finite)
    return {
        "n_obs": float(r_finite.size),
        "mean": float(np.mean(r_finite)) if r_finite.size else float("nan"),
        "std": float(np.std(r_finite, ddof=1)) if r_finite.size > 1 else float("nan"),
        "ann_mean": (
            float(np.mean(r_finite) * periods_per_year) if r_finite.size else float("nan")
        ),
        "ann_vol": (
            float(np.std(r_finite, ddof=1) * np.sqrt(periods_per_year))
            if r_finite.size > 1
            else float("nan")
        ),
        "sharpe": annualized_sharpe(r_finite, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(wealth),
        "hit_rate": hit_rate(r_finite),
        "total_return": float(wealth[-1] - 1.0) if wealth.size else float("nan"),
    }


def quantile_metrics(
    quantile_returns: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    quantile_col: str = "quantile",
    ret_col: str = "q_ret",
    periods_per_year: float = TRADING_DAYS,
) -> pl.DataFrame:
    """Per-quantile Sharpe, drawdown, hit rate, and total return."""
    rows: list[dict[str, float | int]] = []
    for q in sorted(quantile_returns.get_column(quantile_col).unique().to_list()):
        sub = (
            quantile_returns.filter(pl.col(quantile_col) == q)
            .sort(date_col)
            .get_column(ret_col)
            .to_numpy()
        )
        stats = summarize_returns(sub, periods_per_year=periods_per_year)
        rows.append({"quantile": int(q), **stats})
    return pl.DataFrame(rows).sort("quantile")


def portfolio_metrics_bundle(
    returns: pl.DataFrame,
    weights: pl.DataFrame | None = None,
    *,
    ret_col: str = "net_ret",
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
) -> dict[str, float]:
    """Combine return metrics with average turnover (if weights provided)."""
    out = summarize_returns(returns, ret_col=ret_col)
    if weights is not None and not weights.is_empty():
        to = compute_turnover(weights, date_col=date_col, symbol_col=symbol_col)
        out.update(average_turnover(to))
    return out


def metrics_frame(metrics: Mapping[str, float], *, name: str = "portfolio") -> pl.DataFrame:
    """Single-row Polars frame from a metrics dict."""
    return pl.DataFrame({"name": [name], **{k: [v] for k, v in metrics.items()}})
