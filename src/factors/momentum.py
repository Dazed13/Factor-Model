"""Momentum (WML / UMD): prior 12–1 month return, winners minus losers.

Skip the most recent month to avoid short-term reversal microstructure effects.
Signal at month M uses cumulative return from M-12 to M-2.
"""

from __future__ import annotations

import polars as pl

from src.factors.buckets import assign_tercile_labels, long_short_returns, month_end_dates
from src.factors.winsorize import prepare_signal

# Approx trading days: 21 per month
MONTH_DAYS: int = 21
MOM_LOOKBACK: int = 12 * MONTH_DAYS  # ~12 months
MOM_SKIP: int = 1 * MONTH_DAYS  # skip most recent month


def attach_momentum_characteristic(
    df: pl.DataFrame,
    *,
    ret_col: str = "ret",
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    lookback: int = MOM_LOOKBACK,
    skip: int = MOM_SKIP,
    out_col: str = "mom_12_1",
    min_periods: int | None = None,
) -> pl.DataFrame:
    """Cumulative return from t-lookback to t-skip (exclusive of recent skip window).

    Implementation: ``prod(1+r)_{t-lookback:t-skip} - 1`` via log-sum-exp of
    lagged returns, vectorized with rolling windows then shift(skip).
    """
    if lookback <= skip:
        raise ValueError("lookback must exceed skip")
    window = lookback - skip
    min_p = min_periods if min_periods is not None else max(5, window // 2)

    work = df.sort([symbol_col, date_col]).with_columns(
        (1.0 + pl.col(ret_col).fill_null(0.0)).log().alias("_log1p")
    )
    # Rolling sum of log(1+r) over the formation window ending at t, then shift
    # by `skip` so the window ends at t-skip (excludes most recent month).
    work = work.with_columns(
        pl.col("_log1p")
        .rolling_sum(window_size=window, min_periods=min_p)
        .over(symbol_col)
        .alias("_roll_sum")
    )
    return work.with_columns(
        pl.col("_roll_sum")
        .shift(skip)
        .over(symbol_col)
        .exp()
        .sub(1.0)
        .alias(out_col)
    ).drop(["_log1p", "_roll_sum"])


def construct_wml(
    panel: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    ret_col: str = "ret",
    mom_col: str = "mom_12_1",
    weight_col: str = "market_cap",
) -> pl.DataFrame:
    """Winners-Minus-Losers using 30/70 momentum breakpoints."""
    work = panel
    if mom_col not in work.columns:
        work = attach_momentum_characteristic(work, ret_col=ret_col, out_col=mom_col)

    work = assign_tercile_labels(
        work,
        mom_col,
        date_col=date_col,
        out_col="mom_bucket",
        low_label="L",
        mid_label="N",
        high_label="W",
    )
    return long_short_returns(
        work,
        long_mask=pl.col("mom_bucket") == "W",
        short_mask=pl.col("mom_bucket") == "L",
        date_col=date_col,
        ret_col=ret_col,
        weight_col=weight_col if weight_col in work.columns else None,
        weighting="value" if weight_col in work.columns else "equal",
        factor_name="WML",
    )


def construct_illiq_factor(
    panel: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    ret_col: str = "ret",
    signal_col: str = "illiq_signal",
    weight_col: str = "market_cap",
) -> pl.DataFrame:
    """Long illiquid / short liquid using lagged Amihud signal terciles."""
    work = assign_tercile_labels(
        panel,
        signal_col,
        date_col=date_col,
        out_col="illiq_bucket",
        low_label="LIQ",
        mid_label="N",
        high_label="ILLIQ",
    )
    return long_short_returns(
        work,
        long_mask=pl.col("illiq_bucket") == "ILLIQ",
        short_mask=pl.col("illiq_bucket") == "LIQ",
        date_col=date_col,
        ret_col=ret_col,
        weight_col=weight_col if weight_col in work.columns else None,
        weighting="value" if weight_col in work.columns else "equal",
        factor_name="ILLIQ",
    )


def momentum_signal_panel(
    df: pl.DataFrame,
    *,
    winsor: bool = True,
) -> pl.DataFrame:
    out = attach_momentum_characteristic(df)
    return prepare_signal(out, "mom_12_1", winsor=winsor)


def rebalance_month_ends(panel: pl.DataFrame, date_col: str = "trade_date") -> list:
    """Expose month-end calendar for diagnostics / Phase 4."""
    return month_end_dates(panel.get_column(date_col))
