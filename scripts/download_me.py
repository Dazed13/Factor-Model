#!/usr/bin/env python3
"""Build market equity = Bhavcopy close × Yahoo get_shares_full.

Default: attach ME onto Screener book ``as_of_date`` rows and rewrite
``data/processed/fundamentals/fundamentals.parquet`` with BTM filled.

Examples
--------
    # Smoke: fetch shares + ME for a few names, merge into screener book
    python scripts/download_me.py --symbols RELIANCE,TCS,INFY

    # Full universe (symbols from screener_book / Nifty 500)
    python scripts/download_me.py --pause 0.15

    # Reuse cached shares; only recompute ME
    python scripts/download_me.py --shares-from data/raw/fundamentals/shares_outstanding.parquet

    # Month-end ME audit panel only (not merged into book panel)
    python scripts/download_me.py --frequency month-end --no-merge
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ensure_repo_on_path

ensure_repo_on_path()

import polars as pl  # noqa: E402

from src.data.fundamentals import FUNDAMENTAL_COLUMNS, save_fundamentals  # noqa: E402
from src.data.market_equity import (  # noqa: E402
    default_shares_cache_path,
    fetch_shares_history,
    load_bhavcopy_closes,
    load_shares_history,
    market_equity_asof,
    merge_me_into_fundamentals,
    month_end_asof_frame,
    save_market_equity,
    save_shares_history,
)
from src.data.paths import (  # noqa: E402
    PROCESSED_BHAVCOPY_DIR,
    PROCESSED_FUNDAMENTALS_DIR,
    RAW_FUNDAMENTALS_DIR,
    RAW_UNIVERSE_DIR,
    ensure_data_dirs,
)
from src.data.universe import fetch_nifty500_constituents, load_universe_snapshots  # noqa: E402


def _load_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    book_path = args.book if args.book is not None else (
        PROCESSED_FUNDAMENTALS_DIR / "screener_book.parquet"
    )
    if book_path.exists():
        return (
            pl.read_parquet(book_path)
            .get_column("symbol")
            .unique()
            .sort()
            .to_list()
        )

    snaps = load_universe_snapshots(RAW_UNIVERSE_DIR)
    if not snaps.is_empty():
        return snaps.get_column("symbol").unique().sort().to_list()

    logging.info("No local universe / book — fetching current Nifty 500")
    return fetch_nifty500_constituents().get_column("symbol").to_list()


def _load_book(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pl.read_csv(path, try_parse_dates=True)
    else:
        df = pl.read_parquet(path)
    rename = {c: c.strip().lower() for c in df.columns}
    df = df.rename(rename)
    for col in FUNDAMENTAL_COLUMNS:
        if col not in df.columns:
            if col in {"as_of_date", "available_date"}:
                df = df.with_columns(pl.lit(None).cast(pl.Date).alias(col))
            elif col == "symbol":
                raise ValueError(f"{path} missing symbol column")
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    return df.with_columns(
        pl.col("as_of_date").cast(pl.Date),
        pl.col("available_date").cast(pl.Date),
    ).select(FUNDAMENTAL_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Market equity from Bhavcopy close × Yahoo shares"
    )
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--book",
        type=Path,
        default=None,
        help="Book fundamentals panel (default: processed screener_book.parquet)",
    )
    parser.add_argument(
        "--frequency",
        choices=["book", "month-end"],
        default="book",
        help="As-of dates: book panel dates (default) or month-end trade dates",
    )
    parser.add_argument(
        "--shares-from",
        type=Path,
        default=None,
        help="Reuse cached shares Parquet instead of fetching",
    )
    parser.add_argument(
        "--shares-start",
        type=date.fromisoformat,
        default=date(2019, 1, 1),
        help="Yahoo shares history start (default: 2019-01-01)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.15,
        help="Seconds between Yahoo share fetches (default: 0.15)",
    )
    parser.add_argument(
        "--bhavcopy-dir",
        type=Path,
        default=PROCESSED_BHAVCOPY_DIR,
        help="Processed Bhavcopy root (year=YYYY partitions)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Only load Bhavcopy years >= this (default: 2020)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Only load Bhavcopy years <= this (default: 2025)",
    )
    parser.add_argument(
        "--name",
        default="market_equity",
        help="ME audit panel stem (default: market_equity)",
    )
    parser.add_argument(
        "--merged-name",
        default="fundamentals",
        help="Merged fundamentals stem (default: fundamentals)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not overlay ME onto the book panel",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 even if no ME rows",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    ensure_data_dirs()

    symbols = _load_symbols(args)
    if args.limit is not None:
        symbols = symbols[: args.limit]
    logging.info("Building ME for %d symbols (%s)", len(symbols), args.frequency)

    # --- shares ---
    if args.shares_from is not None:
        shares = load_shares_history(args.shares_from)
        logging.info("Loaded %d share rows from %s", shares.height, args.shares_from)
    else:
        cached = default_shares_cache_path()
        if cached.exists() and args.shares_from is None:
            # Still refresh for requested symbols unless --shares-from forced;
            # always fetch to keep cache complete for the run's symbol list.
            pass
        shares = fetch_shares_history(
            symbols,
            start=args.shares_start,
            pause_s=args.pause,
        )
        if shares.is_empty():
            print("No shares history returned from Yahoo — check network / tickers.")
            return 0 if args.allow_empty else 1
        save_shares_history(shares)

    shares = shares.filter(pl.col("symbol").is_in(symbols))
    if shares.is_empty():
        print("Shares panel empty after symbol filter.")
        return 0 if args.allow_empty else 1

    # --- closes ---
    years = list(range(args.start_year, args.end_year + 1))
    # Include prior year so early-2020 as-of can still find a close / shares lag.
    years_load = sorted(set([args.start_year - 1, *years]))
    closes = load_bhavcopy_closes(
        args.bhavcopy_dir,
        symbols=symbols,
        years=years_load,
    )
    if closes.is_empty():
        print(f"No Bhavcopy closes under {args.bhavcopy_dir} for years {years_load}.")
        return 0 if args.allow_empty else 1
    logging.info("Loaded %d close rows", closes.height)

    # --- as-of dates ---
    book_path = args.book or (PROCESSED_FUNDAMENTALS_DIR / "screener_book.parquet")
    book: pl.DataFrame | None = None
    if args.frequency == "book":
        if not book_path.exists():
            print(f"Book panel not found: {book_path}")
            print("Run scripts/download_screener_book.py first, or pass --book.")
            return 1
        book = _load_book(book_path)
        as_of = (
            book.filter(pl.col("symbol").is_in(symbols))
            .select(["as_of_date", "symbol"])
            .unique()
        )
    else:
        as_of = month_end_asof_frame(
            closes,
            start=date(args.start_year, 1, 1),
            end=date(args.end_year, 12, 31),
        )

    logging.info("Computing ME for %d as-of keys", as_of.height)
    me = market_equity_asof(as_of, closes, shares)
    n_me = me.filter(pl.col("market_cap").is_not_null()).height
    logging.info("ME non-null: %d / %d", n_me, me.height)

    if n_me == 0:
        print("No non-null market_cap rows — check shares coverage vs Bhavcopy.")
        return 0 if args.allow_empty else 1

    me_path = save_market_equity(me, name=args.name)
    print(f"Saved ME audit panel → {me_path} ({n_me} non-null of {me.height})")

    if args.no_merge:
        return 0

    if book is None:
        if not book_path.exists():
            print(f"--merge needs a book panel at {book_path} (or pass --book).")
            return 1
        book = _load_book(book_path)

    # Always merge onto the full book panel so --symbols/--limit only
    # refreshes ME for a subset without wiping other names.
    merged = merge_me_into_fundamentals(book, me)
    out = save_fundamentals(merged, name=args.merged_name)
    csv_path = RAW_FUNDAMENTALS_DIR / f"{args.merged_name}.csv"
    merged.write_csv(csv_path)
    n_both = merged.filter(
        pl.col("market_cap").is_not_null() & pl.col("book_value").is_not_null()
    ).height
    print(
        f"Merged ME into book → {out} "
        f"({merged.height} rows, {n_both} with both ME and book)"
    )
    print(f"Also wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
