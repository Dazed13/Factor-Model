#!/usr/bin/env python3
"""Build split/bonus adjustment factors from yfinance and join onto Bhavcopy.

Requires Bhavcopy already present under data/raw/bhavcopy/ (or a processed panel).

Examples
--------
    python scripts/download_adjustments.py --limit-symbols 50
    python scripts/download_adjustments.py --symbols RELIANCE,TCS,INFY
    python scripts/download_adjustments.py --from-processed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ensure_repo_on_path, parse_date

ensure_repo_on_path()

from src.data.bhavcopy import load_bhavcopy_dir  # noqa: E402
from src.data.corporate_actions import attach_yfinance_adjustments  # noqa: E402
from src.data.paths import (  # noqa: E402
    PROCESSED_BHAVCOPY_DIR,
    RAW_BHAVCOPY_DIR,
    RAW_UNIVERSE_DIR,
    ensure_data_dirs,
)
from src.data.store import read_partitioned_parquet, write_partitioned_parquet  # noqa: E402
from src.data.universe import load_universe_snapshots  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attach yfinance Adj Close ratios to NSE Bhavcopy panels"
    )
    parser.add_argument(
        "--from-processed",
        action="store_true",
        help="Load from data/processed/bhavcopy instead of parsing raw zips",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated tickers (default: universe file ∩ panel, else all)",
    )
    parser.add_argument(
        "--limit-symbols",
        type=int,
        default=None,
        help="Cap number of symbols (smoke tests)",
    )
    parser.add_argument("--start", type=parse_date, default=None)
    parser.add_argument("--end", type=parse_date, default=None)
    parser.add_argument(
        "--universe-only",
        action="store_true",
        help="Restrict to symbols in data/raw/universe snapshots",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_data_dirs()

    if args.from_processed:
        bhav = read_partitioned_parquet(PROCESSED_BHAVCOPY_DIR)
    else:
        bhav = load_bhavcopy_dir(RAW_BHAVCOPY_DIR)

    if bhav.is_empty():
        print("No Bhavcopy found. Run scripts/download_bhavcopy.py first.")
        return 1

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.universe_only:
        snaps = load_universe_snapshots(RAW_UNIVERSE_DIR)
        if snaps.is_empty():
            print("No universe snapshots — run scripts/download_universe.py first.")
            return 1
        symbols = snaps.get_column("symbol").unique().to_list()
    else:
        symbols = bhav.get_column("symbol").unique().sort().to_list()

    if args.limit_symbols is not None:
        symbols = symbols[: args.limit_symbols]

    logging.info("Adjusting %d symbols over panel rows=%s", len(symbols), bhav.height)
    # Restrict panel to requested symbols to speed joins
    panel = bhav.filter(bhav["symbol"].is_in(symbols))
    adjusted = attach_yfinance_adjustments(
        panel, symbols=symbols, start=args.start, end=args.end
    )

    paths = write_partitioned_parquet(
        adjusted, PROCESSED_BHAVCOPY_DIR, partition_by="year"
    )
    print(f"Wrote adjusted Bhavcopy partitions ({len(paths)} files) → {PROCESSED_BHAVCOPY_DIR}")
    if "adj_close" in adjusted.columns:
        sample = adjusted.filter(adjusted["adj_factor"] != 1.0).height
        print(f"Rows with adj_factor ≠ 1: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
