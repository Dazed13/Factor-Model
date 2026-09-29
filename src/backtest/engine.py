"""Backtest engine: rebalance calendar, weight propagation, daily PnL.

Signals and universe membership at rebalance $t$ use information available at
or before $t-1$ EOD (characteristics must already be lagged, e.g. Amihud
``illiq_signal``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import polars as pl

from src.backtest.portfolio import (
    Weighting,
    assert_dollar_neutral,
    form_quantile_portfolio,
)
from src.backtest.sorts import SortKind, filter_eligible, lag_signal, rebalance_dates
from src.backtest.turnover import average_turnover, compute_turnover

logger = logging.getLogger(__name__)

RebalanceFreq = Literal["daily", "month_end"]


@dataclass
class BacktestConfig:
    """Knobs for a single long–short characteristic backtest."""

    signal_col: str
    kind: SortKind = "quintile"
    long_high: bool = True
    weighting: Weighting = "equal"
    weight_col: str | None = "market_cap"
    frequency: RebalanceFreq = "month_end"
    ret_col: str = "ret"
    date_col: str = "trade_date"
    symbol_col: str = "symbol"
    # Optional extra lag if the signal column is not already lagged
    ensure_lag: bool = False
    lag_periods: int = 1
    # Proportional one-way transaction cost (e.g. 0.001 = 10 bps)
    tc_bps: float = 0.0
    validate_weights: bool = True


@dataclass
class BacktestResult:
    """Outputs of :func:`run_backtest`."""

    weights: pl.DataFrame
    daily_weights: pl.DataFrame
    returns: pl.DataFrame
    turnover: pl.DataFrame
    config: BacktestConfig
    metrics: dict[str, float] = field(default_factory=dict)


def _formation_panel(
    panel: pl.DataFrame,
    cfg: BacktestConfig,
    *,
    universe: pl.DataFrame | None,
    reb_dates: list[date],
) -> pl.DataFrame:
    """Rows on rebalance dates only, with eligibility filters applied."""
    work = panel
    signal = cfg.signal_col
    if cfg.ensure_lag:
        work = lag_signal(
            work,
            cfg.signal_col,
            symbol_col=cfg.symbol_col,
            date_col=cfg.date_col,
            periods=cfg.lag_periods,
        )
        signal = f"{cfg.signal_col}_lag{cfg.lag_periods}"

    work = filter_eligible(
        work,
        universe=universe,
        date_col=cfg.date_col,
        symbol_col=cfg.symbol_col,
        require_return=False,  # formation uses signal; returns applied when holding
        signal_col=signal,
    )
    work = work.filter(pl.col(cfg.date_col).is_in(reb_dates))
    # Expose the effective signal name for portfolio formation
    if signal != cfg.signal_col:
        work = work.with_columns(pl.col(signal).alias(cfg.signal_col))
    return work


def propagate_weights(
    formation_weights: pl.DataFrame,
    calendar: list[date],
    *,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """Hold formation weights constant between rebalance dates (forward-fill).

    Weight decided at rebalance $t$ earns returns from $t+1$ through the next
    rebalance (inclusive of next rebalance day's open→close via daily return at
    each holding date). Implementation: forward-fill as-of join onto the full
    trading calendar, then shift weights by 1 day so same-day formation does
    not earn same-day return (no lookahead).
    """
    if formation_weights.is_empty() or not calendar:
        return formation_weights.clear()

    cal = pl.DataFrame({date_col: calendar}).with_columns(pl.col(date_col).cast(pl.Date))
    symbols = formation_weights.get_column(symbol_col).unique().to_list()
    grid = pl.DataFrame(
        {
            date_col: calendar * len(symbols),
            symbol_col: [s for s in symbols for _ in calendar],
        }
    ).with_columns(pl.col(date_col).cast(pl.Date))

    w = formation_weights.select(
        [date_col, symbol_col, "weight"]
        + ([c for c in ("quantile", "is_long", "is_short") if c in formation_weights.columns])
    ).sort([symbol_col, date_col])

    # As-of backward join per symbol
    parts: list[pl.DataFrame] = []
    for sym in symbols:
        g = grid.filter(pl.col(symbol_col) == sym).sort(date_col)
        ws = w.filter(pl.col(symbol_col) == sym).sort(date_col)
        if ws.is_empty():
            continue
        joined = g.join_asof(ws.drop(symbol_col), on=date_col, strategy="backward")
        parts.append(joined)

    if not parts:
        return formation_weights.clear()

    daily = pl.concat(parts, how="diagonal_relaxed").sort([symbol_col, date_col])
    # Shift weights by 1 day within symbol: rebalance at t trades for t+1 onward
    daily = daily.with_columns(
        pl.col("weight").shift(1).over(symbol_col).alias("weight")
    ).filter(pl.col("weight").is_not_null())
    return daily


def portfolio_daily_returns(
    daily_weights: pl.DataFrame,
    returns: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    ret_col: str = "ret",
    tc_bps: float = 0.0,
    turnover: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Aggregate $r_{p,t} = \\sum_i w_{i,t} R_{i,t}$, optionally net of costs."""
    panel = daily_weights.join(
        returns.select([date_col, symbol_col, ret_col]),
        on=[date_col, symbol_col],
        how="left",
    )
    pnl = (
        panel.group_by(date_col)
        .agg(
            (pl.col("weight") * pl.col(ret_col)).sum().alias("gross_ret"),
            pl.col("weight").sum().alias("w_sum"),
        )
        .sort(date_col)
    )

    if tc_bps and turnover is not None and not turnover.is_empty():
        # Apply cost on rebalance dates: tc * two-way turnover
        cost = turnover.select(
            [
                date_col,
                (pl.col("turnover_two_way") * (tc_bps / 10_000.0)).alias("tc"),
            ]
        )
        pnl = pnl.join(cost, on=date_col, how="left").with_columns(
            pl.col("tc").fill_null(0.0)
        )
        pnl = pnl.with_columns((pl.col("gross_ret") - pl.col("tc")).alias("net_ret"))
    else:
        pnl = pnl.with_columns(
            pl.col("gross_ret").alias("net_ret"),
            pl.lit(0.0).alias("tc"),
        )

    return pnl.with_columns(
        (1.0 + pl.col("net_ret")).cum_prod().alias("wealth")
    )


def run_backtest(
    panel: pl.DataFrame,
    cfg: BacktestConfig,
    *,
    universe: pl.DataFrame | None = None,
) -> BacktestResult:
    """Run a dollar-neutral quantile long–short backtest."""
    if panel.is_empty():
        empty = panel.clear()
        return BacktestResult(
            weights=empty,
            daily_weights=empty,
            returns=empty,
            turnover=empty,
            config=cfg,
        )

    reb = rebalance_dates(panel, date_col=cfg.date_col, frequency=cfg.frequency)
    formation = _formation_panel(panel, cfg, universe=universe, reb_dates=reb)
    if formation.is_empty():
        logger.warning("No eligible rows on rebalance dates for signal=%s", cfg.signal_col)
        empty = panel.clear()
        return BacktestResult(
            weights=empty,
            daily_weights=empty,
            returns=empty,
            turnover=empty,
            config=cfg,
        )

    weights = form_quantile_portfolio(
        formation,
        cfg.signal_col,
        kind=cfg.kind,
        date_col=cfg.date_col,
        symbol_col=cfg.symbol_col,
        long_high=cfg.long_high,
        weighting=cfg.weighting,
        weight_col=cfg.weight_col,
        validate=cfg.validate_weights,
    )
    if cfg.validate_weights:
        assert_dollar_neutral(weights, date_col=cfg.date_col)

    calendar = panel.get_column(cfg.date_col).unique().sort().to_list()
    daily_w = propagate_weights(
        weights,
        calendar,
        date_col=cfg.date_col,
        symbol_col=cfg.symbol_col,
    )
    turnover = compute_turnover(
        weights, date_col=cfg.date_col, symbol_col=cfg.symbol_col
    )
    rets = portfolio_daily_returns(
        daily_w,
        panel,
        date_col=cfg.date_col,
        symbol_col=cfg.symbol_col,
        ret_col=cfg.ret_col,
        tc_bps=cfg.tc_bps,
        turnover=turnover,
    )

    avg_to = average_turnover(turnover)
    metrics = {
        **avg_to,
        "n_rebals": float(len(reb)),
        "n_days": float(rets.height),
        "total_net_return": float(rets["wealth"][-1] - 1.0) if rets.height else 0.0,
    }

    return BacktestResult(
        weights=weights,
        daily_weights=daily_w,
        returns=rets,
        turnover=turnover,
        config=cfg,
        metrics=metrics,
    )


def run_quantile_spread(
    panel: pl.DataFrame,
    signal_col: str,
    *,
    kind: SortKind = "quintile",
    date_col: str = "trade_date",
    ret_col: str = "ret",
    weighting: Weighting = "equal",
    weight_col: str | None = "market_cap",
) -> pl.DataFrame:
    """Average return by quantile bucket each day (long-only sleeve diagnostics)."""
    from src.backtest.sorts import assign_sorts

    sorted_panel = assign_sorts(panel, signal_col, date_col=date_col, kind=kind)
    base = sorted_panel.filter(
        pl.col("quantile").is_not_null() & pl.col(ret_col).is_not_null()
    )
    if weighting == "value" and weight_col and weight_col in base.columns:
        work = base.with_columns(
            pl.col(weight_col).cast(pl.Float64).fill_null(0).clip(lower_bound=0).alias("_w")
        )
    else:
        work = base.with_columns(pl.lit(1.0).alias("_w"))

    tot = work.group_by([date_col, "quantile"]).agg(pl.col("_w").sum().alias("_tot"))
    work = work.join(tot, on=[date_col, "quantile"], how="left")
    return (
        work.with_columns((pl.col("_w") / pl.col("_tot") * pl.col(ret_col)).alias("_wr"))
        .group_by([date_col, "quantile"])
        .agg(pl.col("_wr").sum().alias("q_ret"))
        .sort([date_col, "quantile"])
    )
