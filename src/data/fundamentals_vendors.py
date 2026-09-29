"""Vendor fundamentals fetchers (EODHD, FMP) → canonical FUNDAMENTAL_COLUMNS.

Environment
-----------
* ``EODHD_API_TOKEN`` — https://eodhd.com (demo token works for AAPL.US etc.)
* ``FMP_API_KEY`` — https://site.financialmodelingprep.com

NSE tickers are requested as ``SYMBOL.NSE`` (EODHD) and ``SYMBOL.NS`` (FMP).
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Iterable

import polars as pl

from src.data.fundamentals import FUNDAMENTAL_COLUMNS, apply_reporting_lag
from src.data.http_util import http_get_json
from src.data.symbols import SymbolMap

logger = logging.getLogger(__name__)

EODHD_BASE = "https://eodhd.com/api"
FMP_BASE = "https://financialmodelingprep.com"


def _empty_fundamentals() -> pl.DataFrame:
    return pl.DataFrame(schema={c: pl.Utf8 for c in FUNDAMENTAL_COLUMNS}).clear()


def _to_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _parse_date(val: Any) -> date | None:
    if val is None or val == "":
        return None
    if isinstance(val, date):
        return val
    try:
        return date.fromisoformat(str(val)[:10])
    except ValueError:
        return None


def to_eodhd_symbol(symbol: str, symbol_map: SymbolMap | None = None) -> str:
    sm = symbol_map or SymbolMap()
    bare = sm.normalize_nse(symbol)
    # Already exchange-qualified?
    upper = symbol.strip().upper()
    if "." in upper and not upper.endswith((".NS", ".NSE", ".BO", ".BSE")):
        return upper
    if upper.endswith((".US", ".LSE", ".CC", ".FOREX", ".INDX")):
        return upper
    return f"{bare}.NSE"


def to_fmp_symbol(symbol: str, symbol_map: SymbolMap | None = None) -> str:
    sm = symbol_map or SymbolMap()
    bare = sm.normalize_nse(symbol)
    upper = symbol.strip().upper()
    if upper.endswith((".NS", ".NSE")):
        return f"{bare}.NS"
    if "." in upper:
        return upper
    return f"{bare}.NS"


# ---------------------------------------------------------------------------
# EODHD
# ---------------------------------------------------------------------------


def fetch_eodhd_fundamentals_snapshot(
    symbols: Iterable[str],
    *,
    api_token: str | None = None,
    as_of: date | None = None,
    lag_months: int = 3,
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Current ME / BTM snapshot from EODHD Fundamentals Highlights + Valuation."""
    token = api_token or os.environ.get("EODHD_API_TOKEN") or os.environ.get("EODHD_API_KEY")
    if not token:
        raise ValueError("EODHD_API_TOKEN (or EODHD_API_KEY) is not set")

    sm = symbol_map or SymbolMap()
    as_of = as_of or date.today()
    rows: list[dict[str, object]] = []

    for raw in symbols:
        bare = sm.normalize_nse(raw)
        ticker = to_eodhd_symbol(raw, sm)
        url = f"{EODHD_BASE}/v1.1/fundamentals/{ticker}"
        try:
            data = http_get_json(
                url,
                params={
                    "api_token": token,
                    "fmt": "json",
                    "filter": "Highlights,Valuation,SharesStats",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("EODHD snapshot failed for %s: %s", ticker, exc)
            continue

        if not isinstance(data, dict) or data.get("Error"):
            logger.warning("EODHD snapshot empty/error for %s: %s", ticker, data)
            continue

        # Multi-filter returns nested sections; single-filter is flat.
        highs = data.get("Highlights") if isinstance(data.get("Highlights"), dict) else data
        val = data.get("Valuation") if isinstance(data.get("Valuation"), dict) else {}
        shares_stats = (
            data.get("SharesStats") if isinstance(data.get("SharesStats"), dict) else {}
        )

        mcap = _to_float(highs.get("MarketCapitalization"))
        book_ps = _to_float(highs.get("BookValue"))
        shares = _to_float(shares_stats.get("SharesOutstanding"))
        pb = _to_float(val.get("PriceBookMRQ")) if isinstance(val, dict) else None

        book: float | None = None
        btm: float | None = None
        if pb and pb > 0:
            btm = 1.0 / pb
        if book_ps is not None and shares is not None:
            book = book_ps * shares
            if btm is None and mcap and mcap > 0:
                btm = book / mcap

        mrq = _parse_date(highs.get("MostRecentQuarter")) or as_of
        rows.append(
            {
                "as_of_date": mrq,
                "symbol": bare,
                "market_cap": mcap,
                "book_value": book,
                "book_to_market": btm,
            }
        )

    if not rows:
        return _empty_fundamentals()
    df = pl.DataFrame(rows)
    df = apply_reporting_lag(df, lag_months=lag_months)
    return df.select(FUNDAMENTAL_COLUMNS)


def fetch_eodhd_fundamentals_quarterly(
    symbols: Iterable[str],
    *,
    api_token: str | None = None,
    lag_months: int = 3,
    symbol_map: SymbolMap | None = None,
    max_periods: int = 8,
) -> pl.DataFrame:
    """Quarterly book equity panel from EODHD balance sheets.

    ``market_cap`` is filled from the snapshot Highlights ME only on the latest
    period when available; older periods keep ``book_value`` and derive BTM when
    ME is present. Prefer joining historical prices for production ME.
    """
    token = api_token or os.environ.get("EODHD_API_TOKEN") or os.environ.get("EODHD_API_KEY")
    if not token:
        raise ValueError("EODHD_API_TOKEN (or EODHD_API_KEY) is not set")

    sm = symbol_map or SymbolMap()
    rows: list[dict[str, object]] = []

    for raw in symbols:
        bare = sm.normalize_nse(raw)
        ticker = to_eodhd_symbol(raw, sm)
        url = f"{EODHD_BASE}/v1.1/fundamentals/{ticker}"
        try:
            data = http_get_json(
                url,
                params={
                    "api_token": token,
                    "fmt": "json",
                    "filter": "Highlights,SharesStats,Financials::Balance_Sheet::quarterly",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("EODHD quarterly failed for %s: %s", ticker, exc)
            continue

        if not isinstance(data, dict):
            continue

        highs = data.get("Highlights") if isinstance(data.get("Highlights"), dict) else {}
        # Filter path may return the quarterly map under several key shapes.
        bs = _extract_eodhd_quarterly_bs(data)
        if not isinstance(bs, dict) or not bs:
            logger.warning("EODHD no quarterly BS for %s", ticker)
            continue

        mcap_now = _to_float(highs.get("MarketCapitalization")) if highs else None

        periods = sorted(bs.keys(), reverse=True)[:max_periods]
        for i, period in enumerate(periods):
            entry = bs[period]
            if not isinstance(entry, dict):
                continue
            as_of = _parse_date(entry.get("date") or entry.get("filing_date") or period)
            if as_of is None:
                continue
            book = _to_float(
                entry.get("totalStockholderEquity")
                or entry.get("commonStockTotalEquity")
                or entry.get("totalEquity")
            )
            if book is None:
                continue
            # Only attach current ME to the most recent filing (honest).
            mcap = mcap_now if i == 0 else None
            btm = (book / mcap) if (mcap and mcap > 0) else None
            rows.append(
                {
                    "as_of_date": as_of,
                    "symbol": bare,
                    "market_cap": mcap,
                    "book_value": book,
                    "book_to_market": btm,
                }
            )

    if not rows:
        return _empty_fundamentals()
    df = pl.DataFrame(rows)
    df = apply_reporting_lag(df, lag_months=lag_months)
    return df.select(FUNDAMENTAL_COLUMNS)


def _extract_eodhd_quarterly_bs(data: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize EODHD balance-sheet quarterly payloads across filter shapes."""
    direct = data.get("Financials::Balance_Sheet::quarterly")
    if isinstance(direct, dict) and direct:
        return direct
    bs = data.get("Financials") or data.get("Balance_Sheet")
    if isinstance(bs, dict) and "Balance_Sheet" in bs:
        bs = bs["Balance_Sheet"]
    if isinstance(bs, dict) and "quarterly" in bs:
        bs = bs["quarterly"]
    if isinstance(bs, dict) and bs and all(
        isinstance(v, dict) for v in list(bs.values())[:3]
    ):
        return bs
    return None


# ---------------------------------------------------------------------------
# FMP
# ---------------------------------------------------------------------------


def fetch_fmp_fundamentals_snapshot(
    symbols: Iterable[str],
    *,
    api_key: str | None = None,
    as_of: date | None = None,
    lag_months: int = 3,
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Current ME / BTM from FMP profile + ratios / key-metrics."""
    key = api_key or os.environ.get("FMP_API_KEY") or os.environ.get("FMP_API_TOKEN")
    if not key:
        raise ValueError("FMP_API_KEY (or FMP_API_TOKEN) is not set")

    sm = symbol_map or SymbolMap()
    as_of = as_of or date.today()
    rows: list[dict[str, object]] = []

    for raw in symbols:
        bare = sm.normalize_nse(raw)
        ticker = to_fmp_symbol(raw, sm)
        try:
            profile = http_get_json(
                f"{FMP_BASE}/stable/profile",
                params={"symbol": ticker, "apikey": key},
            )
            # Fallback to legacy v3 if stable returns empty/error
            if not profile or (isinstance(profile, dict) and profile.get("Error Message")):
                profile = http_get_json(
                    f"{FMP_BASE}/api/v3/profile/{ticker}",
                    params={"apikey": key},
                )
            metrics = http_get_json(
                f"{FMP_BASE}/stable/key-metrics",
                params={"symbol": ticker, "apikey": key, "limit": 1},
            )
            if not metrics or (isinstance(metrics, dict) and metrics.get("Error Message")):
                metrics = http_get_json(
                    f"{FMP_BASE}/api/v3/key-metrics/{ticker}",
                    params={"apikey": key, "limit": 1},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FMP snapshot failed for %s: %s", ticker, exc)
            continue

        if isinstance(profile, dict) and profile.get("Error Message"):
            logger.warning("FMP profile error for %s: %s", ticker, profile.get("Error Message"))
            continue
        if isinstance(profile, list):
            profile = profile[0] if profile else {}
        if not isinstance(profile, dict) or not profile:
            logger.warning("FMP empty profile for %s", ticker)
            continue

        mcap = _to_float(profile.get("marketCap") or profile.get("mktCap"))
        metric0: dict[str, Any] = {}
        if isinstance(metrics, list) and metrics:
            metric0 = metrics[0] if isinstance(metrics[0], dict) else {}
        elif isinstance(metrics, dict) and "Error Message" not in metrics:
            metric0 = metrics

        pb = _to_float(metric0.get("pbRatio") or metric0.get("priceToBookRatio"))
        book = _to_float(metric0.get("bookValuePerShare"))
        shares = _to_float(profile.get("sharesOutstanding"))
        btm: float | None = None
        book_total: float | None = None
        if pb and pb > 0:
            btm = 1.0 / pb
        if book is not None and shares is not None:
            book_total = book * shares
            if btm is None and mcap and mcap > 0:
                btm = book_total / mcap

        rows.append(
            {
                "as_of_date": as_of,
                "symbol": bare,
                "market_cap": mcap,
                "book_value": book_total,
                "book_to_market": btm,
            }
        )

    if not rows:
        return _empty_fundamentals()
    df = pl.DataFrame(rows)
    df = apply_reporting_lag(df, lag_months=lag_months)
    return df.select(FUNDAMENTAL_COLUMNS)


def fetch_fmp_fundamentals_quarterly(
    symbols: Iterable[str],
    *,
    api_key: str | None = None,
    lag_months: int = 3,
    symbol_map: SymbolMap | None = None,
    max_periods: int = 8,
) -> pl.DataFrame:
    """Quarterly book equity from FMP balance-sheet statements."""
    key = api_key or os.environ.get("FMP_API_KEY") or os.environ.get("FMP_API_TOKEN")
    if not key:
        raise ValueError("FMP_API_KEY (or FMP_API_TOKEN) is not set")

    sm = symbol_map or SymbolMap()
    rows: list[dict[str, object]] = []

    for raw in symbols:
        bare = sm.normalize_nse(raw)
        ticker = to_fmp_symbol(raw, sm)
        try:
            bs = http_get_json(
                f"{FMP_BASE}/stable/balance-sheet-statement",
                params={"symbol": ticker, "period": "quarter", "limit": max_periods, "apikey": key},
            )
            if not bs or (isinstance(bs, dict) and bs.get("Error Message")):
                bs = http_get_json(
                    f"{FMP_BASE}/api/v3/balance-sheet-statement/{ticker}",
                    params={"period": "quarter", "limit": max_periods, "apikey": key},
                )
            profile = http_get_json(
                f"{FMP_BASE}/stable/profile",
                params={"symbol": ticker, "apikey": key},
            )
            if not profile or (isinstance(profile, dict) and profile.get("Error Message")):
                profile = http_get_json(
                    f"{FMP_BASE}/api/v3/profile/{ticker}",
                    params={"apikey": key},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FMP quarterly failed for %s: %s", ticker, exc)
            continue

        if isinstance(bs, dict) and bs.get("Error Message"):
            logger.warning("FMP BS error for %s: %s", ticker, bs.get("Error Message"))
            continue
        if not isinstance(bs, list):
            continue

        mcap_now: float | None = None
        if isinstance(profile, list) and profile:
            mcap_now = _to_float(profile[0].get("marketCap") or profile[0].get("mktCap"))
        elif isinstance(profile, dict):
            mcap_now = _to_float(profile.get("marketCap") or profile.get("mktCap"))

        for i, entry in enumerate(bs[:max_periods]):
            if not isinstance(entry, dict):
                continue
            as_of = _parse_date(entry.get("date") or entry.get("fillingDate") or entry.get("acceptedDate"))
            if as_of is None:
                continue
            book = _to_float(
                entry.get("totalStockholdersEquity")
                or entry.get("totalStockholderEquity")
                or entry.get("totalEquity")
            )
            if book is None:
                continue
            mcap = mcap_now if i == 0 else None
            btm = (book / mcap) if (mcap and mcap > 0) else None
            rows.append(
                {
                    "as_of_date": as_of,
                    "symbol": bare,
                    "market_cap": mcap,
                    "book_value": book,
                    "book_to_market": btm,
                }
            )

    if not rows:
        return _empty_fundamentals()
    df = pl.DataFrame(rows)
    df = apply_reporting_lag(df, lag_months=lag_months)
    return df.select(FUNDAMENTAL_COLUMNS)
