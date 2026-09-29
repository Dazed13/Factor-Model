#!/usr/bin/env python3
"""Import India risk-free yields (RBI 91-Day T-Bill / MIBOR).

RBI DBIE has no stable public bulk API, so the usual workflow is:
  1. Export 91-Day T-Bill yields from https://data.rbi.org.in/DBIE/ to CSV
  2. Import with this script

This script can also write a template, or pull a CSV from a URL you provide.

Examples
--------
    python scripts/download_risk_free.py --template
    python scripts/download_risk_free.py --from-csv path/to/rbi_91d.csv
    python scripts/download_risk_free.py --from-url https://example.com/tbill.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ensure_repo_on_path

ensure_repo_on_path()

from src.data.paths import RAW_RISK_FREE_DIR, ensure_data_dirs  # noqa: E402
from src.data.risk_free import (  # noqa: E402
    example_rbi_tbill_template,
    import_risk_free_file,
    import_risk_free_url,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import RBI 91-Day T-Bill / MIBOR yields into the project store"
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--template",
        action="store_true",
        help="Write an example CSV template under data/raw/risk_free/",
    )
    g.add_argument(
        "--from-csv",
        type=Path,
        help="Local CSV with date + yield columns",
    )
    g.add_argument(
        "--from-url",
        type=str,
        help="HTTP(S) URL to a CSV with date + yield columns",
    )
    parser.add_argument(
        "--source",
        choices=["rbi_91d_tbill", "mibor"],
        default="rbi_91d_tbill",
        help="Label stored in the risk-free panel (default: rbi_91d_tbill)",
    )
    parser.add_argument(
        "--name",
        default="risk_free",
        help="Output stem under data/processed/risk_free/ (default: risk_free)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_data_dirs()

    if args.template:
        path = example_rbi_tbill_template()
        print(f"Wrote template → {path}")
        print(
            "Replace yields with a real RBI 91-Day T-Bill series covering your "
            "Bhavcopy window, then re-run with --from-csv."
        )
        print(f"RBI DBIE: https://data.rbi.org.in/DBIE/  (export CSV → {RAW_RISK_FREE_DIR})")
        return 0

    if args.from_csv is not None:
        out = import_risk_free_file(
            args.from_csv, source=args.source, dest_name=args.name
        )
        print(f"Imported {args.from_csv} → {out}")
        return 0

    out = import_risk_free_url(args.from_url, source=args.source, dest_name=args.name)
    print(f"Downloaded + imported → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
