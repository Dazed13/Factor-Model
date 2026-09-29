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


def _silence_yfinance_logs() -> None:
    """Yahoo crumb/SSL warnings are extremely noisy and usually non-fatal."""
    for name in ("yfinance", "peewee", "urllib3", "curl_cffi"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _fast_info_get(fi: object, *keys: str) -> object | None:
    for key in keys:
        try:
            if hasattr(fi, "get"):
                val = fi.get(key)  # type: ignore[call-arg]
            else:
                val = getattr(fi, key, None)
            if val is not None:
                return val
        except Exception:  # noqa: BLE001
            continue
    return None


def fetch_yfinance_fundamentals(
    symbols: Iterable[str],
    *,
    as_of: date | None = None,
    lag_months: int = 3,
    symbol_map: SymbolMap | None = None,
    pause_s: float = 0.05,
) -> pl.DataFrame:
    """Pull market cap (and BTM when available) via yfinance ``fast_info``.

    Prefers ``fast_info`` over ``info`` because crumb/SSL failures often make
    ``Ticker.info`` return empty/401 while quote ``fast_info`` still works.

    This is a *current* snapshot helper for scaffolding and smoke tests.
    Production research should replace or enrich with historical filings
    stored under ``data/raw/fundamentals/``.
    """
    import time

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for fetch_yfinance_fundamentals") from exc

    _silence_yfinance_logs()

    sm = symbol_map or SymbolMap()
    as_of = as_of or date.today()
    bare = [sm.normalize_nse(s) for s in symbols]
    yf_syms = to_yfinance_symbols(bare, sm)

    rows: list[dict[str, object]] = []
    n_ok = 0
    n_fail = 0
    total = len(bare)
    for i, (bare_sym, yf_sym) in enumerate(zip(bare, yf_syms), start=1):
        mcap: float | None = None
        book: float | None = None
        btm: float | None = None
        try:
            # Prefer fast_info — avoids Yahoo crumb/info endpoints that 401.
            fi = yf.Ticker(yf_sym).fast_info
            mcap_raw = _fast_info_get(fi, "marketCap", "market_cap")
            if mcap_raw is not None:
                mcap = float(mcap_raw)

            # BTM is rarely on fast_info; leave null in snapshot mode.
            # Use --mode quarterly (or a vendor CSV) for book equity / HML.
            if mcap is None:
                n_fail += 1
            else:
                n_ok += 1
                rows.append(
                    {
                        "as_of_date": as_of,
                        "symbol": bare_sym,
                        "market_cap": mcap,
                        "book_value": book,
                        "book_to_market": btm,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            logger.warning("yfinance snapshot failed for %s: %s", yf_sym, exc)

        if i % 25 == 0 or i == total:
            logger.info("Fundamentals progress: %d/%d (ok=%d fail=%d)", i, total, n_ok, n_fail)
        if pause_s > 0:
            time.sleep(pause_s)

    logger.info("Snapshot fundamentals done: ok=%d fail=%d", n_ok, n_fail)
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
        else:
            df = df.with_columns(
                pl.col(col).cast(pl.Float64, strict=False).alias(col)
            )
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


def _resolve_shares(ticker: object, info: dict) -> float | None:
    """Best-effort shares outstanding from info or share-count history."""
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    if shares:
        try:
            return float(shares)
        except (TypeError, ValueError):
            pass
    try:
        hist_shares = ticker.get_shares_full(start="2015-01-01")
        if hist_shares is not None and len(hist_shares) > 0:
            return float(hist_shares.dropna().iloc[-1])
    except Exception:  # noqa: BLE001
        pass
    # Derive from marketCap / last price when possible
    mcap = info.get("marketCap")
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    try:
        if mcap and price and float(price) > 0:
            return float(mcap) / float(price)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


def _pick_equity_row(bs: object) -> object | None:
    """Find a book-equity-like row in a yfinance balance-sheet DataFrame."""
    if bs is None or getattr(bs, "empty", True):
        return None
    equity_keys = (
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest",
        "StockholdersEquity",
        "CommonStockEquity",
        "TotalEquityGrossMinorityInterest",
    )
    index_map = {str(i): i for i in bs.index}
    lower_map = {str(i).lower().replace(" ", ""): i for i in bs.index}
    for key in equity_keys:
        if key in index_map:
            return bs.loc[index_map[key]]
        compact = key.lower().replace(" ", "")
        if compact in lower_map:
            return bs.loc[lower_map[compact]]
    # Fuzzy: any index containing both 'stockholder' and 'equity'
    for i in bs.index:
        s = str(i).lower()
        if "equity" in s and ("stockholder" in s or "shareholder" in s or "common stock" in s):
            return bs.loc[i]
    return None


def fetch_yfinance_quarterly_fundamentals(
    symbols: Iterable[str],
    *,
    lag_months: int = 3,
    symbol_map: SymbolMap | None = None,
    max_symbols: int | None = None,
) -> pl.DataFrame:
    """Build a quarterly ME / BTM panel from yfinance filings + price history.

    Method (good enough for pipeline smoke tests; not CMIE-grade PIT):
      * Book equity from quarterly (else annual) balance sheet.
      * Market cap ≈ filing-date close × shares outstanding (best-effort).

    Prefer a vendor fundamentals file for production research.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for fetch_yfinance_quarterly_fundamentals") from exc

    _silence_yfinance_logs()

    sm = symbol_map or SymbolMap()
    bare = [sm.normalize_nse(s) for s in symbols]
    if max_symbols is not None:
        bare = bare[: max_symbols]
    yf_syms = to_yfinance_symbols(bare, sm)

    rows: list[dict[str, object]] = []
    n_ok = 0
    n_fail = 0
    for bare_sym, yf_sym in zip(bare, yf_syms):
        try:
            t = yf.Ticker(yf_sym)
            # Prefer fast_info; .info often 401s on crumb/SSL issues
            try:
                fi = t.fast_info
                info = {
                    "marketCap": _fast_info_get(fi, "marketCap", "market_cap"),
                    "currentPrice": _fast_info_get(
                        fi, "lastPrice", "last_price", "regularMarketPrice"
                    ),
                    "sharesOutstanding": _fast_info_get(fi, "shares", "sharesOutstanding"),
                }
            except Exception:  # noqa: BLE001
                info = {}
            shares = _resolve_shares(t, info)
            bs = t.quarterly_balance_sheet
            book_row = _pick_equity_row(bs)
            if book_row is None:
                book_row = _pick_equity_row(getattr(t, "balance_sheet", None))
            hist = t.history(period="max", auto_adjust=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("quarterly fundamentals failed for %s: %s", yf_sym, exc)
            n_fail += 1
            continue

        if book_row is None or shares is None:
            logger.warning(
                "Insufficient quarterly data for %s (book=%s shares=%s)",
                yf_sym,
                book_row is not None,
                shares,
            )
            n_fail += 1
            continue

        if hist is None or getattr(hist, "empty", True):
            n_fail += 1
            continue

        hist = hist.copy()
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)

        wrote = 0
        for as_of_ts, book_val in book_row.items():
            try:
                as_of = (
                    as_of_ts.date()
                    if hasattr(as_of_ts, "date")
                    else date.fromisoformat(str(as_of_ts)[:10])
                )
            except Exception:  # noqa: BLE001
                continue
            try:
                book = float(book_val)
            except (TypeError, ValueError):
                continue
            if book != book:  # NaN
                continue
            px = hist.loc[: str(as_of)]
            if px.empty:
                continue
            close = float(px["Close"].iloc[-1])
            if close <= 0:
                continue
            mcap = close * float(shares)
            btm = book / mcap if mcap > 0 else None
            rows.append(
                {
                    "as_of_date": as_of,
                    "symbol": bare_sym,
                    "market_cap": mcap,
                    "book_value": book,
                    "book_to_market": btm,
                }
            )
            wrote += 1

        if wrote:
            n_ok += 1
        else:
            n_fail += 1

    logger.info("Quarterly fundamentals: %d symbols ok, %d failed", n_ok, n_fail)
    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 for c in FUNDAMENTAL_COLUMNS}).clear()

    df = (
        pl.DataFrame(rows)
        .unique(subset=["as_of_date", "symbol"])
        .sort(["symbol", "as_of_date"])
    )
    df = apply_reporting_lag(df, lag_months=lag_months)
    return df.select(FUNDAMENTAL_COLUMNS)
