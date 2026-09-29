"""Cleaning transforms and survivorship / liquidation policy.

Survivorship bias policy
------------------------
If a historical constituent ceases trading or is suspended, **retain** the
time series and mark subsequent returns as NaN (liquidation / unavailable).
Never silently drop history or restrict panels to today's survivors.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Iterable, Sequence

import polars as pl

from src.data.bhavcopy import ALLOWED_SERIES
from src.data.symbols import SymbolMap

logger = logging.getLogger(__name__)

CLEAN_FLAG_COLUMNS: list[str] = [
    "is_suspended",
    "is_missing_ohlc",
    "is_duplicate",
    "is_liquidated",
]


def cast_bhavcopy_types(df: pl.DataFrame) -> pl.DataFrame:
    """Enforce canonical dtypes for a bhavcopy-like panel."""
    casts: list[pl.Expr] = []
    if "trade_date" in df.columns:
        casts.append(pl.col("trade_date").cast(pl.Date))
    if "symbol" in df.columns:
        casts.append(pl.col("symbol").cast(pl.Utf8).str.strip_chars().str.to_uppercase())
    if "series" in df.columns:
        casts.append(pl.col("series").cast(pl.Utf8).str.strip_chars().str.to_uppercase())

    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "last",
        "prev_close",
        "tottrdqty",
        "tottrdval",
        "adj_factor",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_last",
        "adj_prev_close",
    ]
    for c in float_cols:
        if c in df.columns:
            casts.append(pl.col(c).cast(pl.Float64, strict=False))

    return df.with_columns(casts) if casts else df


def filter_equity_series(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only ``EQ`` / ``BE`` rows."""
    if "series" not in df.columns:
        return df
    return df.filter(pl.col("series").is_in(list(ALLOWED_SERIES)))


def deduplicate_bhavcopy(df: pl.DataFrame) -> pl.DataFrame:
    """Drop duplicate (trade_date, symbol), preferring EQ over BE and denser rows.

    Adds transient ``is_duplicate`` on dropped rows is not retained; survivors
    are unique on ``(trade_date, symbol)``.
    """
    if df.is_empty():
        return df

    work = df.with_columns(
        pl.when(pl.col("series") == "EQ")
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("_series_rank"),
        (
            pl.col("close").is_not_null().cast(pl.Int8)
            + pl.col("tottrdval").is_not_null().cast(pl.Int8)
        ).alias("_density"),
    )
    return (
        work.sort(["trade_date", "symbol", "_series_rank", "_density"], descending=[False, False, False, True])
        .unique(subset=["trade_date", "symbol"], keep="first")
        .drop(["_series_rank", "_density"])
    )


def flag_data_quality(df: pl.DataFrame) -> pl.DataFrame:
    """Attach suspension / missing-OHLC flags (rows are retained)."""
    missing_ohlc = (
        pl.col("open").is_null()
        | pl.col("high").is_null()
        | pl.col("low").is_null()
        | pl.col("close").is_null()
        | (pl.col("close") <= 0)
    )
    suspended = (
        pl.col("tottrdqty").fill_null(0) <= 0
    ) | (
        pl.col("tottrdval").fill_null(0) <= 0
    ) | missing_ohlc

    return df.with_columns(
        suspended.alias("is_suspended"),
        missing_ohlc.alias("is_missing_ohlc"),
        pl.lit(False).alias("is_liquidated"),
    )


def trading_calendar_from_panel(
    df: pl.DataFrame,
    *,
    date_col: str = "trade_date",
) -> list[date]:
    """Infer the exchange calendar as sorted unique dates present in the panel."""
    if df.is_empty() or date_col not in df.columns:
        return []
    return df.get_column(date_col).unique().sort().to_list()


def build_weekday_calendar(start: date, end: date) -> list[date]:
    """Simple Mon–Fri calendar (NSE holiday calendar can replace this later)."""
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def apply_liquidation_policy(
    df: pl.DataFrame,
    *,
    calendar: Sequence[date] | None = None,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    symbols: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Expand panel onto a trading calendar; mark post-delist gaps as liquidated.

    For each symbol:
      * Keep all historical rows (never drop).
      * After the last observed trade date, subsequent calendar dates are
        retained with null prices and ``is_liquidated=True``.
      * Interior gaps (suspension / missing session) are also retained with
        null fills and ``is_suspended=True`` when not liquidated.

    Point-in-time universe membership for portfolio eligibility is handled
    separately in ``universe.py`` — this function only preserves price history.
    """
    if df.is_empty():
        return df

    work = cast_bhavcopy_types(df)
    if "is_suspended" not in work.columns:
        work = flag_data_quality(work)
    if "is_liquidated" not in work.columns:
        work = work.with_columns(pl.lit(False).alias("is_liquidated"))

    cal = list(calendar) if calendar is not None else trading_calendar_from_panel(work, date_col=date_col)
    if not cal:
        return work

    syms = (
        list(symbols)
        if symbols is not None
        else work.get_column(symbol_col).unique().sort().to_list()
    )

    # Last observed trade per symbol (any non-null close)
    last_trade = (
        work.filter(pl.col("close").is_not_null())
        .group_by(symbol_col)
        .agg(pl.col(date_col).max().alias("_last_trade"))
    )

    grid = pl.DataFrame(
        {
            date_col: cal * len(syms),
            symbol_col: [s for s in syms for _ in cal],
        }
    ).with_columns(
        pl.col(date_col).cast(pl.Date),
        pl.col(symbol_col).cast(pl.Utf8),
    )

    merged = grid.join(work, on=[date_col, symbol_col], how="left")
    merged = merged.join(last_trade, on=symbol_col, how="left")

    # Liquidated: calendar date strictly after last observed trade
    liquidated = pl.col(date_col) > pl.col("_last_trade")
    # Suspended gap: on calendar, no close, but not yet liquidated
    gap_suspended = pl.col("close").is_null() & ~liquidated

    merged = merged.with_columns(
        liquidated.fill_null(False).alias("is_liquidated"),
        (
            pl.col("is_suspended").fill_null(False) | gap_suspended | liquidated.fill_null(False)
        ).alias("is_suspended"),
        pl.col("is_missing_ohlc").fill_null(pl.col("close").is_null()),
    ).drop("_last_trade")

    return merged.sort([symbol_col, date_col])


def clean_bhavcopy(
    df: pl.DataFrame,
    *,
    symbol_map: SymbolMap | None = None,
    apply_calendar: bool = False,
    calendar: Sequence[date] | None = None,
) -> pl.DataFrame:
    """Full clean: normalize → filter series → dedupe → quality flags → optional liquidation grid."""
    sm = symbol_map or SymbolMap()
    if df.is_empty():
        return df

    out = cast_bhavcopy_types(df)
    if "symbol" in out.columns:
        out = sm.normalize_frame(out, "symbol")
    out = filter_equity_series(out)
    out = deduplicate_bhavcopy(out)
    out = flag_data_quality(out)

    if apply_calendar:
        out = apply_liquidation_policy(out, calendar=calendar)

    logger.info(
        "clean_bhavcopy: rows=%s symbols=%s suspended=%s",
        out.height,
        out.get_column("symbol").n_unique() if out.height else 0,
        int(out["is_suspended"].sum()) if out.height and "is_suspended" in out.columns else 0,
    )
    return out
