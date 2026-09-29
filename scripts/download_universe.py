#!/usr/bin/env python3
"""Download / archive the current Nifty 500 constituent list.

True point-in-time history requires repeating this over time (or supplying
dated archives). This script fetches today's public list.

Examples
--------
    python scripts/download_universe.py
    python scripts/download_universe.py --as-of 2024-06-30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ensure_repo_on_path, parse_date

ensure_repo_on_path()

from src.data.paths import ensure_data_dirs  # noqa: E402
from src.data.universe import fetch_nifty500_constituents, save_universe_snapshot  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Nifty 500 constituents")
    parser.add_argument(
        "--as-of",
        type=parse_date,
        default=None,
        help="Label snapshot date (default: today). Does not time-travel the source file.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_data_dirs()

    df = fetch_nifty500_constituents(as_of=args.as_of)
    path = save_universe_snapshot(df)
    print(f"Saved {df.height} constituents → {path}")
    print("Also wrote CSV under data/raw/universe/")
    print(
        "Note: public CSV is a *current* list. For PIT membership, re-run "
        "monthly and keep dated files, or supply historical archives."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
