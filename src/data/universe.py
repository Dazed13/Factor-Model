"""Nifty 500 universe membership (point-in-time aware).

True historical PIT membership requires archived constituent snapshots.
This module:
  1. Fetches the current Nifty 500 list from NSE / niftyindices.
  2. Loads dated snapshot CSVs from ``data/raw/universe/``.
  3. Builds a membership panel without look-ahead when snapshots are provided.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Iterable

import polars as pl
import requests

from src.data.paths import RAW_UNIVERSE_DIR, PROCESSED_UNIVERSE_DIR, ensure_data_dirs
from src.data.symbols import SymbolMap

logger = logging.getLogger(__name__)

NIFTY500_URLS: tuple[str, ...] = (
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
)

_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/",
}

UNIVERSE_SCHEMA: list[str] = ["as_of_date", "symbol", "company_name", "industry", "isin"]


def _parse_constituent_csv(text: str, as_of: date, symbol_map: SymbolMap) -> pl.DataFrame:
    """Parse NSE / niftyindices constituent CSV into canonical schema."""
    raw = pl.read_csv(StringIO(text), infer_schema_length=5000)
    colmap: dict[str, str] = {}
    lower = {c.lower().strip(): c for c in raw.columns}
    for key, dest in (
        ("symbol", "symbol"),
        ("company name", "company_name"),
        ("company", "company_name"),
        ("industry", "industry"),
        ("isin code", "isin"),
        ("isin", "isin"),
    ):
        if key in lower and dest not in colmap.values():
            colmap[lower[key]] = dest

    if "symbol" not in colmap.values():
        raise ValueError(f"Could not find Symbol column in constituent CSV: {raw.columns}")

    df = raw.rename(colmap)
    keep = [c for c in ("symbol", "company_name", "industry", "isin") if c in df.columns]
    df = df.select(keep)
    for missing in ("company_name", "industry", "isin"):
        if missing not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(missing))

    df = symbol_map.normalize_frame(df, "symbol")
    return (
        df.with_columns(pl.lit(as_of).alias("as_of_date"))
        .select(UNIVERSE_SCHEMA)
        .unique(subset=["as_of_date", "symbol"])
        .sort(["symbol"])
    )


def fetch_nifty500_constituents(
    as_of: date | None = None,
    *,
    symbol_map: SymbolMap | None = None,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Download the current Nifty 500 constituent list.

    Notes
    -----
    The public CSV is a *current* snapshot. For historical PIT membership,
    archive dated files under ``data/raw/universe/nifty500_YYYY-MM-DD.csv``.
    """
    sm = symbol_map or SymbolMap()
    as_of = as_of or date.today()
    sess = session or requests.Session()
    sess.headers.update(_DEFAULT_HEADERS)

    last_err: Exception | None = None
    for url in NIFTY500_URLS:
        try:
            # Warm-up cookie jar for archives.nseindia.com
            if "nseindia" in url:
                sess.get("https://www.nseindia.com", timeout=timeout)
            resp = sess.get(url, timeout=timeout)
            resp.raise_for_status()
            return _parse_constituent_csv(resp.text, as_of, sm)
        except Exception as exc:  # noqa: BLE001 — try next mirror
            last_err = exc
            logger.warning("Failed to fetch Nifty 500 from %s: %s", url, exc)

    raise RuntimeError(f"Unable to fetch Nifty 500 constituents: {last_err}")


def save_universe_snapshot(df: pl.DataFrame, directory: Path | None = None) -> Path:
    """Persist a constituent snapshot as snappy Parquet + dated CSV archive."""
    ensure_data_dirs()
    raw_dir = directory or RAW_UNIVERSE_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    as_of = df.select(pl.col("as_of_date").min()).item()
    if isinstance(as_of, datetime):
        as_of = as_of.date()
    stem = f"nifty500_{as_of.isoformat()}"
    csv_path = raw_dir / f"{stem}.csv"
    parquet_path = PROCESSED_UNIVERSE_DIR / f"{stem}.parquet"
    PROCESSED_UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    # Raw archive kept as CSV only for human audit of source snapshot.
    df.write_csv(csv_path)
    df.write_parquet(parquet_path, compression="snappy")
    return parquet_path


def load_universe_snapshots(
    directory: Path | None = None,
    *,
    symbol_map: SymbolMap | None = None,
) -> pl.DataFrame:
    """Load all dated ``nifty500_*.csv`` / ``*.parquet`` snapshots into a panel."""
    sm = symbol_map or SymbolMap()
    raw_dir = directory or RAW_UNIVERSE_DIR
    frames: list[pl.DataFrame] = []

    if raw_dir.exists():
        for path in sorted(raw_dir.glob("nifty500_*.csv")):
            as_of = _as_of_from_name(path.stem)
            text = path.read_text(encoding="utf-8")
            frames.append(_parse_constituent_csv(text, as_of, sm))
        for path in sorted(raw_dir.glob("nifty500_*.parquet")):
            frames.append(sm.normalize_frame(pl.read_parquet(path), "symbol"))

    if PROCESSED_UNIVERSE_DIR.exists():
        for path in sorted(PROCESSED_UNIVERSE_DIR.glob("nifty500_*.parquet")):
            frames.append(sm.normalize_frame(pl.read_parquet(path), "symbol"))

    if not frames:
        return pl.DataFrame(schema={c: pl.Utf8 for c in UNIVERSE_SCHEMA}).with_columns(
            pl.col("as_of_date").cast(pl.Date)
        )

    return pl.concat(frames, how="diagonal_relaxed").unique(
        subset=["as_of_date", "symbol"]
    ).sort(["as_of_date", "symbol"])


def point_in_time_membership(
    snapshots: pl.DataFrame,
    as_of: date,
) -> pl.DataFrame:
    """Return constituents known as of ``as_of`` without look-ahead.

    Uses the latest snapshot with ``as_of_date <= as_of``. If none exists,
    returns an empty frame (caller must supply historical archives).
    """
    if snapshots.is_empty():
        return snapshots.clear()

    eligible = snapshots.filter(pl.col("as_of_date") <= as_of)
    if eligible.is_empty():
        return snapshots.clear()

    latest = eligible.select(pl.col("as_of_date").max()).item()
    return eligible.filter(pl.col("as_of_date") == latest)


def membership_on_dates(
    snapshots: pl.DataFrame,
    dates: Iterable[date],
) -> pl.DataFrame:
    """Expand PIT membership for each date in ``dates``."""
    rows: list[pl.DataFrame] = []
    for d in dates:
        m = point_in_time_membership(snapshots, d)
        if m.is_empty():
            continue
        rows.append(m.with_columns(pl.lit(d).alias("trade_date")))
    if not rows:
        return pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "as_of_date": pl.Date,
                "symbol": pl.Utf8,
                "company_name": pl.Utf8,
                "industry": pl.Utf8,
                "isin": pl.Utf8,
            }
        )
    return pl.concat(rows, how="vertical_relaxed").sort(["trade_date", "symbol"])


def _as_of_from_name(stem: str) -> date:
    """Parse ``nifty500_YYYY-MM-DD`` stem."""
    part = stem.replace("nifty500_", "")
    return date.fromisoformat(part)
