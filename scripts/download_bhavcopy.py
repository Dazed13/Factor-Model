#!/usr/bin/env python3
"""Download NSE equity Bhavcopy zips for an inclusive date range.

Examples
--------
    python scripts/download_bhavcopy.py --start 2020-01-01 --end 2024-12-31
    python scripts/download_bhavcopy.py -s 2023-01-01 -e 2023-12-31 --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Allow running as ``python scripts/download_bhavcopy.py`` from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.bhavcopy import download_bhavcopy_range  # noqa: E402
from src.data.paths import RAW_BHAVCOPY_DIR, ensure_data_dirs  # noqa: E402


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download NSE CM Bhavcopy archives into data/raw/bhavcopy/"
    )
    parser.add_argument(
        "-s",
        "--start",
        type=_parse_date,
        required=True,
        help="Start date (YYYY-MM-DD), inclusive",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=_parse_date,
        required=True,
        help="End date (YYYY-MM-DD), inclusive",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help=f"Destination directory (default: {RAW_BHAVCOPY_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the zip already exists locally",
    )
    parser.add_argument(
        "--include-weekends",
        action="store_true",
        help="Also attempt Sat/Sun (usually 404; off by default)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args(argv)

    if args.end < args.start:
        parser.error("--end must be on or after --start")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ensure_data_dirs()
    dest = args.out_dir or RAW_BHAVCOPY_DIR
    dest.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Downloading Bhavcopy %s → %s into %s (skip_if_present=%s)",
        args.start,
        args.end,
        dest,
        not args.force,
    )

    paths = download_bhavcopy_range(
        args.start,
        args.end,
        dest_dir=dest,
        skip_weekends=not args.include_weekends,
        skip_if_present=not args.force,
    )

    # Count trading weekdays attempted vs successes
    from datetime import timedelta

    attempted = 0
    d = args.start
    while d <= args.end:
        if args.include_weekends or d.weekday() < 5:
            attempted += 1
        d += timedelta(days=1)

    print(
        f"Done: {len(paths)} files available in {dest} "
        f"(attempted ~{attempted} weekdays; holidays/weekends may be missing)."
    )
    if paths:
        print(f"Example: {paths[0].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
