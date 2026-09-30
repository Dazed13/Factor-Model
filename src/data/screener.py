"""Screener.in book-equity scraper (Equity Capital + Reserves).

Public company pages expose annual (and occasionally half-year) balance-sheet
tables in ₹ crore. This module pulls those rows into the canonical fundamentals
schema with ``book_value`` in INR and null ``market_cap`` / ``book_to_market``
(ME is filled separately from prices or a vendor).

Respect screener.in's rate limits; prefer cached HTML for large universes.
"""

from __future__ import annotations

import calendar
import logging
import re
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import polars as pl

from src.data.fundamentals import (
    FUNDAMENTAL_COLUMNS,
    apply_reporting_lag,
    save_fundamentals,
)
from src.data.http_util import http_get_text
from src.data.paths import RAW_FUNDAMENTALS_DIR, ensure_data_dirs
from src.data.symbols import SymbolMap

logger = logging.getLogger(__name__)

SCREENER_BASE = "https://www.screener.in"
# Screener balance-sheet figures are in ₹ crore.
CRORE_TO_INR = 10_000_000.0

_MONTH_MAP: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_PERIOD_RE = re.compile(
    r"^(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?P<year>\d{4})$",
    re.IGNORECASE,
)


class _DataTableParser(HTMLParser):
    """Extract the first ``table.data-table`` inside ``#balance-sheet``."""

    def __init__(self) -> None:
        super().__init__()
        self._in_section = False
        self._in_table = False
        self._in_th = False
        self._in_td = False
        self._in_tr = False
        self._section_depth = 0
        self.headers: list[str] = []
        self.rows: list[list[str]] = []
        self._cur_row: list[str] = []
        self._cur_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        ad = {k: (v or "") for k, v in attrs}
        if tag == "section" and ad.get("id") == "balance-sheet":
            self._in_section = True
            self._section_depth = 1
            return
        if self._in_section and tag == "section":
            self._section_depth += 1
        if not self._in_section:
            return
        if tag == "table" and "data-table" in ad.get("class", "").split():
            # Only the first data-table in the section.
            if not self.headers and not self.rows:
                self._in_table = True
            return
        if not self._in_table:
            return
        if tag == "tr":
            self._in_tr = True
            self._cur_row = []
        elif tag == "th":
            self._in_th = True
            self._cur_text = []
        elif tag == "td":
            self._in_td = True
            self._cur_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "section" and self._in_section:
            self._section_depth -= 1
            if self._section_depth <= 0:
                self._in_section = False
                self._in_table = False
            return
        if not self._in_table:
            return
        if tag == "th" and self._in_th:
            self._in_th = False
            self.headers.append(_clean_label("".join(self._cur_text)))
        elif tag == "td" and self._in_td:
            self._in_td = False
            self._cur_row.append("".join(self._cur_text).strip())
        elif tag == "tr" and self._in_tr:
            self._in_tr = False
            if self._cur_row:
                self.rows.append(self._cur_row)
        elif tag == "table":
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_th or self._in_td:
            self._cur_text.append(data)


def _clean_label(text: str) -> str:
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Expandable rows append a trailing "+"
    text = text.rstrip("+").strip()
    return text


def parse_screener_period(header: str) -> date | None:
    """Parse Screener column headers like ``Mar 2020`` → month-end date."""
    h = header.strip()
    if not h or h.upper() == "TTM":
        return None
    m = _PERIOD_RE.match(h)
    if not m:
        return None
    month = _MONTH_MAP[m.group("mon").lower()]
    year = int(m.group("year"))
    day = calendar.monthrange(year, month)[1]
    return date(year, month, day)


def parse_screener_number(text: str) -> float | None:
    """Parse a Screener numeric cell (commas / blanks / dashes)."""
    s = text.strip().replace(",", "")
    if not s or s in {"-", "—", "–", "NA", "N/A"}:
        return None
    # Percentages are not used for book equity rows.
    if s.endswith("%"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_balance_sheet_table(html: str) -> dict[str, dict[date, float]]:
    """Parse ``#balance-sheet`` into ``{row_label: {as_of: value_crore}}``."""
    parser = _DataTableParser()
    parser.feed(html)
    if not parser.headers:
        return {}

    # Headers: ['', 'Mar 2015', ...] or ['Mar 2015', ...] depending on empty th.
    period_headers = parser.headers
    # Align: first header cell is often blank (row label column).
    if period_headers and period_headers[0] == "":
        period_headers = period_headers[1:]

    periods: list[date | None] = [parse_screener_period(h) for h in period_headers]
    out: dict[str, dict[date, float]] = {}
    for row in parser.rows:
        if not row:
            continue
        label = _clean_label(row[0])
        if not label or label.lower() == "raw pdf":
            continue
        values = row[1:]
        series: dict[date, float] = {}
        for i, cell in enumerate(values):
            if i >= len(periods):
                break
            as_of = periods[i]
            if as_of is None:
                continue
            num = parse_screener_number(cell)
            if num is None:
                continue
            series[as_of] = num
        if series:
            out[label] = series
    return out


def book_equity_crore_from_balance_sheet(
    table: dict[str, dict[date, float]],
) -> dict[date, float]:
    """Book equity (₹ crore) = Equity Capital + Reserves when both present."""
    equity = table.get("Equity Capital") or {}
    reserves = table.get("Reserves") or {}
    dates = set(equity) | set(reserves)
    out: dict[date, float] = {}
    for d in sorted(dates):
        eq = equity.get(d)
        res = reserves.get(d)
        if eq is None and res is None:
            continue
        out[d] = float(eq or 0.0) + float(res or 0.0)
    return out


def screener_company_url(symbol: str, *, consolidated: bool = True) -> str:
    bare = SymbolMap().normalize_nse(symbol)
    if consolidated:
        return f"{SCREENER_BASE}/company/{bare}/consolidated/"
    return f"{SCREENER_BASE}/company/{bare}/"


def fetch_screener_html(
    symbol: str,
    *,
    consolidated: bool = True,
    cache_dir: Path | None = None,
    timeout: float = 45.0,
    force: bool = False,
) -> str:
    """Fetch company HTML, optionally caching under ``cache_dir``."""
    bare = SymbolMap().normalize_nse(symbol)
    suffix = "consolidated" if consolidated else "standalone"
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{bare}_{suffix}.html"
        if path.exists() and not force and path.stat().st_size > 0:
            return path.read_text(encoding="utf-8", errors="replace")

    url = screener_company_url(bare, consolidated=consolidated)
    html = http_get_text(url, timeout=timeout)
    if cache_dir is not None:
        path = cache_dir / f"{bare}_{suffix}.html"
        path.write_text(html, encoding="utf-8")
    return html


def fetch_screener_book_equity(
    symbols: Iterable[str],
    *,
    start_year: int = 2020,
    end_year: int = 2025,
    lag_months: int = 3,
    consolidated: bool = True,
    pause_s: float = 0.75,
    timeout: float = 45.0,
    cache_dir: Path | None = None,
    force: bool = False,
    symbol_map: SymbolMap | None = None,
    fallback_standalone: bool = True,
) -> pl.DataFrame:
    """Download Screener book equity for ``symbols`` into FUNDAMENTAL_COLUMNS.

    ``market_cap`` and ``book_to_market`` are left null — merge ME separately.
    ``book_value`` is absolute INR (crore × 1e7).
    """
    sm = symbol_map or SymbolMap()
    bare_syms = [sm.normalize_nse(s) for s in symbols]
    rows: list[dict[str, object]] = []
    n_ok = 0
    n_fail = 0

    for i, bare in enumerate(bare_syms):
        try:
            html = fetch_screener_html(
                bare,
                consolidated=consolidated,
                cache_dir=cache_dir,
                timeout=timeout,
                force=force,
            )
            table = parse_balance_sheet_table(html)
            book_cr = book_equity_crore_from_balance_sheet(table)
            if not book_cr and fallback_standalone and consolidated:
                html = fetch_screener_html(
                    bare,
                    consolidated=False,
                    cache_dir=cache_dir,
                    timeout=timeout,
                    force=force,
                )
                table = parse_balance_sheet_table(html)
                book_cr = book_equity_crore_from_balance_sheet(table)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Screener book fetch failed for %s: %s", bare, exc)
            n_fail += 1
            if pause_s > 0 and i + 1 < len(bare_syms):
                time.sleep(pause_s)
            continue

        wrote = 0
        for as_of, book_cr_val in book_cr.items():
            if as_of.year < start_year or as_of.year > end_year:
                continue
            rows.append(
                {
                    "as_of_date": as_of,
                    "symbol": bare,
                    "market_cap": None,
                    "book_value": float(book_cr_val) * CRORE_TO_INR,
                    "book_to_market": None,
                }
            )
            wrote += 1

        if wrote:
            n_ok += 1
        else:
            logger.warning(
                "No Screener book rows in %d–%d for %s", start_year, end_year, bare
            )
            n_fail += 1

        if pause_s > 0 and i + 1 < len(bare_syms):
            time.sleep(pause_s)

    logger.info("Screener book equity: %d symbols ok, %d failed/empty", n_ok, n_fail)
    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 for c in FUNDAMENTAL_COLUMNS}).clear()

    df = (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("as_of_date").cast(pl.Date),
            pl.col("market_cap").cast(pl.Float64),
            pl.col("book_value").cast(pl.Float64),
            pl.col("book_to_market").cast(pl.Float64),
        )
        .unique(subset=["as_of_date", "symbol"])
        .sort(["symbol", "as_of_date"])
    )
    df = apply_reporting_lag(df, lag_months=lag_months)
    return df.select(FUNDAMENTAL_COLUMNS)


def merge_book_into_fundamentals(
    base: pl.DataFrame,
    book: pl.DataFrame,
) -> pl.DataFrame:
    """Overlay Screener ``book_value`` onto an existing fundamentals panel.

    Matching keys: ``symbol`` + ``as_of_date``. Screener book wins on overlap;
    base ``market_cap`` is preserved. ``book_to_market`` is recomputed when both
    sides are non-null.
    """
    if book.is_empty():
        return base.select(FUNDAMENTAL_COLUMNS) if not base.is_empty() else book
    if base.is_empty():
        return book.select(FUNDAMENTAL_COLUMNS)

    keys = ["symbol", "as_of_date"]
    book_slim = book.select(
        [*keys, "book_value", "available_date"]
    ).rename(
        {
            "book_value": "_bv_s",
            "available_date": "_ad_s",
        }
    )
    joined = base.join(book_slim, on=keys, how="left")
    joined = joined.with_columns(
        pl.coalesce([pl.col("_bv_s"), pl.col("book_value")]).alias("book_value"),
        pl.when(pl.col("_bv_s").is_not_null())
        .then(pl.col("_ad_s"))
        .otherwise(pl.col("available_date"))
        .alias("available_date"),
    ).drop(["_bv_s", "_ad_s"])

    only_book = book.join(base.select(keys), on=keys, how="anti")
    out = pl.concat([joined, only_book], how="diagonal_relaxed")
    out = out.with_columns(
        pl.when(
            pl.col("book_value").is_not_null()
            & pl.col("market_cap").is_not_null()
            & (pl.col("market_cap") > 0)
        )
        .then(pl.col("book_value") / pl.col("market_cap"))
        .otherwise(pl.col("book_to_market"))
        .alias("book_to_market")
    )
    return (
        out.select(FUNDAMENTAL_COLUMNS)
        .unique(subset=keys)
        .sort(["symbol", "as_of_date"])
    )


def default_screener_cache_dir() -> Path:
    ensure_data_dirs()
    path = RAW_FUNDAMENTALS_DIR / "screener_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_screener_book(df: pl.DataFrame, name: str = "screener_book") -> Path:
    """Persist book panel (processed Parquet + raw CSV)."""
    parquet = save_fundamentals(df, name=name)
    csv_path = RAW_FUNDAMENTALS_DIR / f"{name}.csv"
    df.write_csv(csv_path)
    logger.info("Wrote Screener book panel → %s and %s", parquet, csv_path)
    return parquet
