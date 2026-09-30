#!/usr/bin/env python3
"""Download book equity (Equity Capital + Reserves) from screener.in.

Fills the book side of the fundamentals panel for Nifty 500 (or explicit
tickers). Market cap stays null — merge ME from prices/vendors separately, or
pass ``--merge-into`` to overlay book onto an existing fundamentals file.

Examples
--------
    python scripts/download_screener_book.py --symbols RELIANCE,TCS,INFY
    python scripts/download_screener_book.py --start-year 2020 --end-year 2025
    python scripts/download_screener_book.py --limit 20 --pause 1.0
    python scripts/download_screener_book.py --merge-into data/processed/fundamentals/fundamentals.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ensure_repo_on_path

ensure_repo_on_path()

import polars as pl  # noqa: E402

from src.data.fundamentals import FUNDAMENTAL_COLUMNS, save_fundamentals  # noqa: E402
from src.data.paths import RAW_FUNDAMENTALS_DIR, RAW_UNIVERSE_DIR, ensure_data_dirs  # noqa: E402
from src.data.screener import (  # noqa: E402
    default_screener_cache_dir,
    fetch_screener_book_equity,
    merge_book_into_fundamentals,
    save_screener_book,
)
from src.data.universe import fetch_nifty500_constituents, load_universe_snapshots  # noqa: E402


def _load_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    snaps = load_universe_snapshots(RAW_UNIVERSE_DIR)
    if not snaps.is_empty():
        return snaps.get_column("symbol").unique().sort().to_list()

    logging.info("No local universe snapshot — fetching current Nifty 500")
    u = fetch_nifty500_constituents()
    return u.get_column("symbol").to_list()


def _load_fundamentals(path: Path) -> pl.DataFrame:
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
        description="Download Screener.in book equity for NSE symbols"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated bare NSE tickers (default: Nifty 500 universe file)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N symbols")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--lag-months",
        type=int,
        default=3,
        help="Reporting lag before available_date (default: 3)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.75,
        help="Seconds between company page fetches (default: 0.75)",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Prefer standalone pages (default: consolidated, with standalone fallback)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read/write HTML cache under data/raw/fundamentals/screener_cache/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cached HTML exists",
    )
    parser.add_argument(
        "--name",
        default="screener_book",
        help="Output stem (default: screener_book)",
    )
    parser.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="Existing fundamentals Parquet/CSV to overlay book_value onto",
    )
    parser.add_argument(
        "--merged-name",
        default="fundamentals",
        help="Stem when writing --merge-into result (default: fundamentals)",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 even if no rows",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_data_dirs()

    symbols = _load_symbols(args)
    if args.limit is not None:
        symbols = symbols[: args.limit]
    logging.info(
        "Fetching Screener book equity for %d symbols (%d–%d)",
        len(symbols),
        args.start_year,
        args.end_year,
    )

    cache_dir = None if args.no_cache else default_screener_cache_dir()
    book = fetch_screener_book_equity(
        symbols,
        start_year=args.start_year,
        end_year=args.end_year,
        lag_months=args.lag_months,
        consolidated=not args.standalone,
        pause_s=args.pause,
        cache_dir=cache_dir,
        force=args.force,
    )

    if book.is_empty():
        print("No Screener book rows returned — check network / tickers / rate limits.")
        return 0 if args.allow_empty else 1

    parquet = save_screener_book(book, name=args.name)
    n_sym = book.get_column("symbol").n_unique()
    print(
        f"Saved {book.height} book rows ({n_sym} symbols) → {parquet} "
        f"(and {RAW_FUNDAMENTALS_DIR / f'{args.name}.csv'})"
    )
    print(
        "Note: market_cap / book_to_market are null. "
        "Fill ME separately, or use --merge-into an existing fundamentals file."
    )

    if args.merge_into is not None:
        base = _load_fundamentals(args.merge_into)
        merged = merge_book_into_fundamentals(base, book)
        out = save_fundamentals(merged, name=args.merged_name)
        csv_path = RAW_FUNDAMENTALS_DIR / f"{args.merged_name}.csv"
        merged.write_csv(csv_path)
        print(f"Merged book into {args.merge_into} → {out} ({merged.height} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
