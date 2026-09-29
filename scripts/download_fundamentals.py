#!/usr/bin/env python3
"""Download fundamentals (market cap / book-to-market) via yfinance.

Modes
-----
* ``snapshot`` — current ``Ticker.info`` (fast; not historical PIT)
* ``quarterly`` — quarterly balance-sheet book equity + price×shares ME
  (falls back to snapshot if quarterly returns nothing)

Examples
--------
    python scripts/download_fundamentals.py --mode snapshot --limit 50
    python scripts/download_fundamentals.py --mode quarterly --limit 100
    python scripts/download_fundamentals.py --symbols RELIANCE,TCS,INFY --mode quarterly
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ensure_repo_on_path

ensure_repo_on_path()

from src.data.fundamentals import (  # noqa: E402
    fetch_yfinance_fundamentals,
    fetch_yfinance_quarterly_fundamentals,
    save_fundamentals,
)
from src.data.paths import RAW_FUNDAMENTALS_DIR, RAW_UNIVERSE_DIR, ensure_data_dirs  # noqa: E402
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download NSE fundamentals via yfinance")
    parser.add_argument(
        "--mode",
        choices=["snapshot", "quarterly"],
        default="quarterly",
        help="snapshot=current info; quarterly=filing-linked panel (default)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated bare NSE tickers (default: Nifty 500 universe file)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only first N symbols (useful for smoke tests)",
    )
    parser.add_argument(
        "--lag-months",
        type=int,
        default=3,
        help="Reporting lag before available_date (default: 3)",
    )
    parser.add_argument(
        "--name",
        default="fundamentals",
        help="Output stem (default: fundamentals)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not fall back to snapshot if quarterly returns empty",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit 0 even if no rows (orchestrator-friendly)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # yfinance is extremely noisy at DEBUG
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)

    ensure_data_dirs()

    symbols = _load_symbols(args)
    if args.limit is not None:
        symbols = symbols[: args.limit]
    logging.info("Fetching fundamentals for %d symbols (%s)", len(symbols), args.mode)

    used_mode = args.mode
    if args.mode == "snapshot":
        df = fetch_yfinance_fundamentals(symbols, lag_months=args.lag_months)
    else:
        df = fetch_yfinance_quarterly_fundamentals(
            symbols, lag_months=args.lag_months, max_symbols=None
        )
        if df.is_empty() and not args.no_fallback:
            logging.warning(
                "Quarterly fundamentals empty — falling back to snapshot mode"
            )
            df = fetch_yfinance_fundamentals(symbols, lag_months=args.lag_months)
            used_mode = "snapshot (fallback)"

    if df.is_empty():
        print("No fundamentals rows returned — check network / tickers / Yahoo SSL.")
        print(
            "Tip: retry with --mode snapshot, or continue the pipeline with "
            "--skip-fundamentals (Amihud/WML still work)."
        )
        return 0 if args.allow_empty else 1

    parquet = save_fundamentals(df, name=args.name)
    csv_path = RAW_FUNDAMENTALS_DIR / f"{args.name}.csv"
    df.write_csv(csv_path)
    print(f"Saved {df.height} rows via {used_mode} → {parquet}")
    print(f"Also wrote {csv_path}")
    print(
        "Caveat: yfinance sharesOutstanding is often point-in-time current; "
        "prefer a vendor fundamentals feed for production SMB/HML research."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
