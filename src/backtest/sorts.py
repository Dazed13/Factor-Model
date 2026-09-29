"""Cross-sectional quintile / decile sorts (no lookahead on signals).

At rebalance date $t$, rank on a characteristic that must already be lagged
(e.g. Amihud ``illiq_signal`` which is ``shift(1)`` of the rolling mean).
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from src.factors.buckets import assign_quantile_buckets, month_end_dates

SortKind = Literal["quintile", "decile"]

N_BUCKETS: dict[SortKind, int] = {"quintile": 5, "decile": 10}


def assign_sorts(
    df: pl.DataFrame,
    signal_col: str,
    *,
    date_col: str = "trade_date",
    kind: SortKind = "quintile",
    out_col: str = "quantile",
    ascending: bool = True,
) -> pl.DataFrame:
    """Assign quantile ranks 1..N within each date.

    Bucket 1 = lowest signal if ``ascending`` else highest.
    """
    n = N_BUCKETS[kind]
    return assign_quantile_buckets(
        df,
        signal_col,
        date_col=date_col,
        n=n,
        out_col=out_col,
        ascending=ascending,
    )


def lag_signal(
    df: pl.DataFrame,
    signal_col: str,
    *,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    out_col: str | None = None,
    periods: int = 1,
) -> pl.DataFrame:
    """Explicitly lag a raw characteristic before sorting (lookahead guard)."""
    dest = out_col or f"{signal_col}_lag{periods}"
    return df.sort([symbol_col, date_col]).with_columns(
        pl.col(signal_col).shift(periods).over(symbol_col).alias(dest)
    )


def rebalance_dates(
    df: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    frequency: Literal["daily", "month_end"] = "month_end",
) -> list:
    """Trading dates used for portfolio formation."""
    dates = df.get_column(date_col).unique().sort()
    if frequency == "daily":
        return dates.to_list()
    return month_end_dates(dates)


def filter_eligible(
    df: pl.DataFrame,
    *,
    universe: pl.DataFrame | None = None,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    require_return: bool = True,
    ret_col: str = "ret",
    signal_col: str | None = None,
) -> pl.DataFrame:
    """Point-in-time eligibility: optional universe ∩ non-null signal/return.

    ``universe`` should contain ``trade_date`` (or ``as_of_date``) and ``symbol``.
    If only ``as_of_date`` is present, caller should expand membership first.
    """
    work = df
    if universe is not None and not universe.is_empty():
        u_cols = set(universe.columns)
        if date_col in u_cols and symbol_col in u_cols:
            keys = universe.select([date_col, symbol_col]).unique()
            work = work.join(keys, on=[date_col, symbol_col], how="inner")
        elif "as_of_date" in u_cols and symbol_col in u_cols:
            # Fallback: keep symbols that appear in any snapshot (still not ideal PIT)
            syms = universe.get_column(symbol_col).unique()
            work = work.filter(pl.col(symbol_col).is_in(syms.to_list()))

    if require_return and ret_col in work.columns:
        work = work.filter(pl.col(ret_col).is_not_null())
    if signal_col is not None:
        work = work.filter(pl.col(signal_col).is_not_null())
    return work
