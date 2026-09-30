"""Market equity from NSE Bhavcopy closes × Yahoo shares outstanding.

Uses **unadjusted** Bhavcopy ``close`` (as-traded) times the latest
``get_shares_full`` observation on or before each as-of date. Do not pair
``adj_close`` with raw share counts — splits would double-count.

ME is typically attached onto Screener book ``as_of_date`` rows so the Phase-2
asof join keeps a single fundamentals panel (book + ME + BTM).
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from src.data.fundamentals import FUNDAMENTAL_COLUMNS, save_fundamentals
from src.data.paths import (
    PROCESSED_BHAVCOPY_DIR,
    PROCESSED_FUNDAMENTALS_DIR,
    RAW_FUNDAMENTALS_DIR,
    ensure_data_dirs,
)
from src.data.store import read_partitioned_parquet
from src.data.symbols import SymbolMap, to_yfinance_symbols

logger = logging.getLogger(__name__)

SHARES_COLUMNS: list[str] = ["as_of_date", "symbol", "shares"]
ME_COLUMNS: list[str] = [
    "as_of_date",
    "symbol",
    "close",
    "shares",
    "market_cap",
    "price_date",
    "shares_date",
]


def _silence_yfinance_logs() -> None:
    for name in ("yfinance", "peewee", "urllib3", "curl_cffi"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def default_shares_cache_path() -> Path:
    ensure_data_dirs()
    return RAW_FUNDAMENTALS_DIR / "shares_outstanding.parquet"


def fetch_shares_history(
    symbols: Iterable[str],
    *,
    start: date | str = date(2019, 1, 1),
    pause_s: float = 0.15,
    symbol_map: SymbolMap | None = None,
    max_symbols: int | None = None,
) -> pl.DataFrame:
    """Pull Yahoo ``get_shares_full`` history → ``(as_of_date, symbol, shares)``."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for fetch_shares_history") from exc

    _silence_yfinance_logs()
    sm = symbol_map or SymbolMap()
    bare = [sm.normalize_nse(s) for s in symbols]
    if max_symbols is not None:
        bare = bare[:max_symbols]
    yf_syms = to_yfinance_symbols(bare, sm)
    start_s = start.isoformat() if isinstance(start, date) else str(start)[:10]

    rows: list[dict[str, object]] = []
    n_ok = 0
    n_fail = 0
    for i, (bare_sym, yf_sym) in enumerate(zip(bare, yf_syms)):
        try:
            series = yf.Ticker(yf_sym).get_shares_full(start=start_s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_shares_full failed for %s: %s", yf_sym, exc)
            n_fail += 1
            if pause_s > 0 and i + 1 < len(bare):
                time.sleep(pause_s)
            continue

        if series is None or len(series) == 0:
            logger.warning("Empty shares history for %s", yf_sym)
            n_fail += 1
            if pause_s > 0 and i + 1 < len(bare):
                time.sleep(pause_s)
            continue

        wrote = 0
        for ts, shares_val in series.items():
            try:
                as_of = ts.date() if hasattr(ts, "date") else date.fromisoformat(str(ts)[:10])
            except Exception:  # noqa: BLE001
                continue
            try:
                shares = float(shares_val)
            except (TypeError, ValueError):
                continue
            if shares != shares or shares <= 0:
                continue
            rows.append(
                {
                    "as_of_date": as_of,
                    "symbol": bare_sym,
                    "shares": shares,
                }
            )
            wrote += 1

        if wrote:
            n_ok += 1
        else:
            n_fail += 1
        if pause_s > 0 and i + 1 < len(bare):
            time.sleep(pause_s)

    logger.info("Shares history: %d symbols ok, %d failed/empty", n_ok, n_fail)
    if not rows:
        return pl.DataFrame(
            schema={
                "as_of_date": pl.Date,
                "symbol": pl.Utf8,
                "shares": pl.Float64,
            }
        ).clear()

    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("as_of_date").cast(pl.Date),
            pl.col("shares").cast(pl.Float64),
        )
        .unique(subset=["as_of_date", "symbol"], keep="last")
        .sort(["symbol", "as_of_date"])
        .select(SHARES_COLUMNS)
    )


def save_shares_history(
    df: pl.DataFrame,
    path: Path | None = None,
    *,
    merge_existing: bool = True,
) -> Path:
    """Persist shares panel to raw Parquet (optionally union with prior cache)."""
    ensure_data_dirs()
    out = path or default_shares_cache_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if merge_existing and out.exists() and not df.is_empty():
        prior = load_shares_history(out)
        if not prior.is_empty():
            df = (
                pl.concat([prior, df], how="diagonal_relaxed")
                .unique(subset=["as_of_date", "symbol"], keep="last")
                .sort(["symbol", "as_of_date"])
            )
    df.write_parquet(out, compression="snappy")
    logger.info("Wrote shares history → %s (%d rows)", out, df.height)
    return out


def load_shares_history(path: Path | None = None) -> pl.DataFrame:
    """Load cached shares panel."""
    p = path or default_shares_cache_path()
    if not p.exists():
        return pl.DataFrame(
            schema={
                "as_of_date": pl.Date,
                "symbol": pl.Utf8,
                "shares": pl.Float64,
            }
        ).clear()
    df = pl.read_parquet(p)
    return df.with_columns(
        pl.col("as_of_date").cast(pl.Date),
        pl.col("shares").cast(pl.Float64),
    ).select(SHARES_COLUMNS)


def load_bhavcopy_closes(
    processed_dir: Path | None = None,
    *,
    symbols: Sequence[str] | None = None,
    years: Sequence[int] | None = None,
    price_col: str = "close",
) -> pl.DataFrame:
    """Load non-null Bhavcopy closes (default: unadjusted ``close``)."""
    root = processed_dir or PROCESSED_BHAVCOPY_DIR
    cols = ["trade_date", "symbol", price_col]
    df = read_partitioned_parquet(root, years=years, columns=cols)
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "symbol": pl.Utf8,
                "close": pl.Float64,
            }
        ).clear()

    sm = SymbolMap()
    df = sm.normalize_frame(df, "symbol")
    if price_col != "close":
        df = df.rename({price_col: "close"})
    df = (
        df.with_columns(
            pl.col("trade_date").cast(pl.Date),
            pl.col("close").cast(pl.Float64),
        )
        .filter(pl.col("close").is_not_null() & (pl.col("close") > 0))
        .unique(subset=["trade_date", "symbol"], keep="last")
        .sort(["symbol", "trade_date"])
    )
    if symbols is not None:
        want = {sm.normalize_nse(s) for s in symbols}
        df = df.filter(pl.col("symbol").is_in(sorted(want)))
    return df.select(["trade_date", "symbol", "close"])


def month_end_asof_frame(
    closes: pl.DataFrame,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """Distinct month-end trade dates per symbol from a close panel."""
    if closes.is_empty():
        return pl.DataFrame(schema={"as_of_date": pl.Date, "symbol": pl.Utf8}).clear()

    work = closes.with_columns(
        pl.col("trade_date").dt.truncate("1mo").alias("_month")
    )
    ends = (
        work.group_by(["symbol", "_month"])
        .agg(pl.col("trade_date").max().alias("as_of_date"))
        .select(["as_of_date", "symbol"])
        .sort(["symbol", "as_of_date"])
    )
    if start is not None:
        ends = ends.filter(pl.col("as_of_date") >= start)
    if end is not None:
        ends = ends.filter(pl.col("as_of_date") <= end)
    return ends


def market_equity_asof(
    as_of: pl.DataFrame,
    closes: pl.DataFrame,
    shares: pl.DataFrame,
) -> pl.DataFrame:
    """For each ``(symbol, as_of_date)``, attach last close and shares ≤ as_of.

    Returns rows with ``market_cap = close * shares`` plus audit columns
    ``price_date`` / ``shares_date``.
    """
    if as_of.is_empty():
        return pl.DataFrame(schema={c: pl.Utf8 for c in ME_COLUMNS}).clear()

    keys = as_of.select(
        pl.col("as_of_date").cast(pl.Date),
        pl.col("symbol").cast(pl.Utf8),
    ).unique().sort(["symbol", "as_of_date"])

    px = closes.rename({"trade_date": "price_date"}).sort(["symbol", "price_date"])
    sh = shares.rename({"as_of_date": "shares_date"}).sort(["symbol", "shares_date"])

    with_px = keys.join_asof(
        px,
        left_on="as_of_date",
        right_on="price_date",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    )
    with_sh = with_px.sort(["symbol", "as_of_date"]).join_asof(
        sh,
        left_on="as_of_date",
        right_on="shares_date",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    )
    out = with_sh.with_columns(
        pl.when(
            pl.col("close").is_not_null()
            & pl.col("shares").is_not_null()
            & (pl.col("close") > 0)
            & (pl.col("shares") > 0)
        )
        .then(pl.col("close") * pl.col("shares"))
        .otherwise(None)
        .alias("market_cap")
    )
    return out.select(ME_COLUMNS).sort(["symbol", "as_of_date"])


def me_to_fundamentals_frame(
    me: pl.DataFrame,
    *,
    available_date: pl.Expr | None = None,
) -> pl.DataFrame:
    """Map an ME panel into FUNDAMENTAL_COLUMNS (book fields null)."""
    if me.is_empty():
        return pl.DataFrame(schema={c: pl.Utf8 for c in FUNDAMENTAL_COLUMNS}).clear()

    df = me.select(
        [
            pl.col("as_of_date"),
            pl.col("symbol"),
            pl.col("market_cap"),
            pl.lit(None).cast(pl.Float64).alias("book_value"),
            pl.lit(None).cast(pl.Float64).alias("book_to_market"),
        ]
    )
    # ME is known at as_of (EOD price); no reporting lag.
    if available_date is None:
        df = df.with_columns(pl.col("as_of_date").alias("available_date"))
    else:
        df = df.with_columns(available_date.alias("available_date"))
    return df.select(FUNDAMENTAL_COLUMNS)


def merge_me_into_fundamentals(
    base: pl.DataFrame,
    me: pl.DataFrame,
) -> pl.DataFrame:
    """Overlay ``market_cap`` onto an existing fundamentals panel (e.g. Screener book).

    Matching keys: ``symbol`` + ``as_of_date``. Recomputes ``book_to_market``
    when both book and ME are present. Preserves base ``available_date`` /
    ``book_value`` (does not shorten the book reporting lag).

    ``me`` may be an ME audit panel (with ``market_cap``) or a fundamentals frame.
    """
    if me.is_empty():
        return base.select(FUNDAMENTAL_COLUMNS) if not base.is_empty() else me
    if base.is_empty():
        return me_to_fundamentals_frame(me) if "close" in me.columns else me.select(
            FUNDAMENTAL_COLUMNS
        )

    keys = ["symbol", "as_of_date"]
    me_slim = me.select([*keys, "market_cap"]).unique(subset=keys)
    joined = base.join(
        me_slim.rename({"market_cap": "_me"}),
        on=keys,
        how="left",
    ).with_columns(
        pl.coalesce([pl.col("_me"), pl.col("market_cap")]).alias("market_cap")
    ).drop("_me")

    only_me = me_slim.join(base.select(keys), on=keys, how="anti")
    if only_me.is_empty():
        out = joined
    else:
        extra = only_me.with_columns(
            pl.col("as_of_date").alias("available_date"),
            pl.lit(None).cast(pl.Float64).alias("book_value"),
            pl.lit(None).cast(pl.Float64).alias("book_to_market"),
        ).select(FUNDAMENTAL_COLUMNS)
        out = pl.concat([joined, extra], how="diagonal_relaxed")

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


def build_market_equity_panel(
    as_of: pl.DataFrame,
    *,
    closes: pl.DataFrame | None = None,
    shares: pl.DataFrame | None = None,
    processed_bhavcopy_dir: Path | None = None,
    shares_path: Path | None = None,
    symbols: Sequence[str] | None = None,
    years: Sequence[int] | None = None,
) -> pl.DataFrame:
    """End-to-end ME panel for the given as-of keys."""
    if closes is None:
        closes = load_bhavcopy_closes(
            processed_bhavcopy_dir,
            symbols=symbols,
            years=years,
        )
    if shares is None:
        shares = load_shares_history(shares_path)
    return market_equity_asof(as_of, closes, shares)


def save_market_equity(
    df: pl.DataFrame,
    name: str = "market_equity",
    *,
    merge_existing: bool = True,
) -> Path:
    """Write ME audit panel (close × shares) to processed Parquet + raw CSV."""
    ensure_data_dirs()
    PROCESSED_FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_FUNDAMENTALS_DIR / f"{name}.parquet"
    if merge_existing and out.exists() and not df.is_empty():
        prior = pl.read_parquet(out)
        df = (
            pl.concat([prior, df], how="diagonal_relaxed")
            .unique(subset=["as_of_date", "symbol"], keep="last")
            .sort(["symbol", "as_of_date"])
        )
    df.write_parquet(out, compression="snappy")
    csv_path = RAW_FUNDAMENTALS_DIR / f"{name}.csv"
    df.write_csv(csv_path)
    logger.info("Wrote market equity → %s and %s", out, csv_path)
    return out
