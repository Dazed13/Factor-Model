"""Fundamental snapshots (market cap, book equity) with reporting lags.

Phase-1 source: ``yfinance`` info / quarterly balance-sheet fields, lagged by
``lag_months`` (default 3) before they become eligible for month-``M`` sorts.
Persist panels as snappy Parquet under ``data/processed/fundamentals/``.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Iterable

import polars as pl

from src.data.paths import (
    PROCESSED_FUNDAMENTALS_DIR,
    RAW_FUNDAMENTALS_DIR,
    ensure_data_dirs,
)
from src.data.symbols import SymbolMap, to_yfinance_symbols

logger = logging.getLogger(__name__)

FUNDAMENTAL_COLUMNS: list[str] = [
    "as_of_date",
    "available_date",
    "symbol",
    "market_cap",
    "book_value",
    "book_to_market",
]


def lag_available_date(as_of: date, lag_months: int = 3) -> date:
    """Shift a fundamental ``as_of`` date forward by ``lag_months`` (no look-ahead)."""
    if lag_months < 0:
        raise ValueError("lag_months must be >= 0")
    month = as_of.month - 1 + lag_months
    year = as_of.year + month // 12
    month = month % 12 + 1
    # Clamp day for month-end
    day = min(as_of.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    return (nxt - date(year, month, 1)).days


def apply_reporting_lag(
    fundamentals: pl.DataFrame,
    *,
    lag_months: int = 3,
    as_of_col: str = "as_of_date",
) -> pl.DataFrame:
    """Attach ``available_date = as_of_date + lag_months``."""
    dates = fundamentals.get_column(as_of_col).to_list()
    available = [lag_available_date(d, lag_months) for d in dates]
    return fundamentals.with_columns(pl.Series("available_date", available))


def point_in_time_fundamentals(
    fundamentals: pl.DataFrame,
    as_of: date,
    *,
    available_col: str = "available_date",
) -> pl.DataFrame:
    """Latest fundamental row per symbol with ``available_date <= as_of``."""
    eligible = fundamentals.filter(pl.col(available_col) <= as_of)
    if eligible.is_empty():
        return eligible
    return (
        eligible.sort(["symbol", available_col])
        .group_by("symbol", maintain_order=True)
        .tail(1)
    )


def fetch_yfinance_fundamentals(
    symbols: Iterable[str],
    *,
    as_of: date | None = None,
    lag_months: int = 3,
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Pull marketCap / bookValue from yfinance ``Ticker.info`` (snapshot).

    This is a *current* snapshot helper for scaffolding and smoke tests.
    Production research should replace or enrich with historical filings
    (quarterly balance sheets) stored under ``data/raw/fundamentals/``.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for fetch_yfinance_fundamentals") from exc

    sm = symbol_map or SymbolMap()
    as_of = as_of or date.today()
    bare = [sm.normalize_nse(s) for s in symbols]
    yf_syms = to_yfinance_symbols(bare, sm)

    rows: list[dict[str, object]] = []
    for bare_sym, yf_sym in zip(bare, yf_syms):
        try:
            info = yf.Ticker(yf_sym).info or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance info failed for %s: %s", yf_sym, exc)
            continue
        mcap = info.get("marketCap")
        book = info.get("bookValue")
        # bookValue from Yahoo is often per-share; prefer book_to_market from
        # priceBook if present, else bookValue / (marketCap/shares) approximation.
        pb = info.get("priceToBook")
        if pb and pb > 0:
            btm = 1.0 / float(pb)
        elif mcap and book and mcap > 0:
            # Heuristic: treat bookValue as per-share and scale by sharesOutstanding
            shares = info.get("sharesOutstanding")
            if shares:
                btm = (float(book) * float(shares)) / float(mcap)
            else:
                btm = None
        else:
            btm = None
        rows.append(
            {
                "as_of_date": as_of,
                "symbol": bare_sym,
                "market_cap": float(mcap) if mcap is not None else None,
                "book_value": float(book) if book is not None else None,
                "book_to_market": float(btm) if btm is not None else None,
            }
        )

    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 for c in FUNDAMENTAL_COLUMNS}).clear()

    df = pl.DataFrame(rows)
    df = apply_reporting_lag(df, lag_months=lag_months)
    return df.select(FUNDAMENTAL_COLUMNS)


def load_fundamentals_csv(
    path: str | Path,
    *,
    lag_months: int = 3,
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Load a user-supplied fundamentals CSV and apply reporting lag.

    Expected columns (case-insensitive): ``as_of_date``, ``symbol``,
    ``market_cap``, and either ``book_to_market`` or ``book_value``.
    """
    sm = symbol_map or SymbolMap()
    df = pl.read_csv(path, try_parse_dates=True)
    rename = {c: c.strip().lower() for c in df.columns}
    df = df.rename(rename)
    required = {"as_of_date", "symbol"}
    if not required.issubset(df.columns):
        raise ValueError(f"Fundamentals CSV missing {required - set(df.columns)}")
    df = sm.normalize_frame(df, "symbol")
    df = df.with_columns(pl.col("as_of_date").cast(pl.Date))
    for col in ("market_cap", "book_value", "book_to_market"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    if (
        df.get_column("book_to_market").null_count() == df.height
        and "book_value" in df.columns
        and "market_cap" in df.columns
    ):
        df = df.with_columns(
            (pl.col("book_value") / pl.col("market_cap")).alias("book_to_market")
        )
    df = apply_reporting_lag(df, lag_months=lag_months)
    return df.select(FUNDAMENTAL_COLUMNS)


def save_fundamentals(df: pl.DataFrame, name: str = "fundamentals") -> Path:
    """Write fundamentals panel to processed Parquet (snappy)."""
    ensure_data_dirs()
    PROCESSED_FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_FUNDAMENTALS_DIR / f"{name}.parquet"
    df.write_parquet(out, compression="snappy")
    return out
