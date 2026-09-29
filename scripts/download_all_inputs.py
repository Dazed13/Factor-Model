#!/usr/bin/env python3
"""Orchestrate download of all Phase-1 input datasets for a date window.

Order
-----
1. Bhavcopy (required)
2. Nifty 500 universe (recommended)
3. Risk-free template or CSV import (required for excess returns)
4. Fundamentals via yfinance (optional but needed for SMB/HML)
5. Corporate-action adjustments via yfinance (recommended)

Examples
--------
    python scripts/download_all_inputs.py \\
        --start 2020-01-01 --end 2025-12-31 \\
        --skip-bhavcopy \\
        --universe-only-adjustments \\
        --fundamentals-mode snapshot
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ensure_repo_on_path, parse_date

ensure_repo_on_path()

from src.data.paths import RAW_RISK_FREE_DIR  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent


def _run(cmd: list[str], *, optional: bool = False) -> bool:
    logging.info("→ %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        if optional:
            logging.warning(
                "Optional step failed (exit %s); continuing. Cmd: %s",
                result.returncode,
                " ".join(cmd),
            )
            return False
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download all market-data inputs")
    parser.add_argument("-s", "--start", type=parse_date, required=True)
    parser.add_argument("-e", "--end", type=parse_date, required=True)
    parser.add_argument(
        "--skip-bhavcopy",
        action="store_true",
        help="Skip Bhavcopy download (use existing raw files)",
    )
    parser.add_argument("--skip-universe", action="store_true")
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--skip-adjustments", action="store_true")
    parser.add_argument(
        "--risk-free-csv",
        type=Path,
        default=None,
        help="If set, import this RBI/MIBOR CSV; else use existing raw CSV or write template",
    )
    parser.add_argument(
        "--fundamentals-mode",
        choices=["snapshot", "quarterly"],
        default="snapshot",
        help="Default snapshot (more reliable). quarterly falls back to snapshot if empty.",
    )
    parser.add_argument("--fundamentals-limit", type=int, default=None)
    parser.add_argument("--adjustments-limit", type=int, default=None)
    parser.add_argument(
        "--universe-only-adjustments",
        action="store_true",
        help="Only adjust Nifty 500 names (much faster)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    py = sys.executable
    vflag = ["-v"] if args.verbose else []

    if not args.skip_bhavcopy:
        _run(
            [
                py,
                str(SCRIPTS / "download_bhavcopy.py"),
                "--start",
                args.start.isoformat(),
                "--end",
                args.end.isoformat(),
                *vflag,
            ]
        )

    if not args.skip_universe:
        _run([py, str(SCRIPTS / "download_universe.py"), *vflag])

    existing_rf = sorted(RAW_RISK_FREE_DIR.glob("*.csv"))
    existing_rf = [p for p in existing_rf if "template" not in p.name.lower()]
    if args.risk_free_csv is not None:
        _run(
            [
                py,
                str(SCRIPTS / "download_risk_free.py"),
                "--from-csv",
                str(args.risk_free_csv),
                *vflag,
            ]
        )
    elif existing_rf:
        logging.info(
            "Using existing risk-free CSV(s) in %s: %s (skipping template)",
            RAW_RISK_FREE_DIR,
            ", ".join(p.name for p in existing_rf),
        )
    else:
        _run([py, str(SCRIPTS / "download_risk_free.py"), "--template", *vflag])
        logging.warning(
            "Risk-free template written only. Place a normalized date,yield CSV in "
            "data/raw/risk_free/ (e.g. from RBI DBIE)."
        )

    if not args.skip_fundamentals:
        cmd = [
            py,
            str(SCRIPTS / "download_fundamentals.py"),
            "--mode",
            args.fundamentals_mode,
            "--allow-empty",
            *vflag,
        ]
        if args.fundamentals_limit is not None:
            cmd += ["--limit", str(args.fundamentals_limit)]
        _run(cmd, optional=True)

    if not args.skip_adjustments:
        cmd = [
            py,
            str(SCRIPTS / "download_adjustments.py"),
            "--start",
            args.start.isoformat(),
            "--end",
            args.end.isoformat(),
            *vflag,
        ]
        if args.universe_only_adjustments:
            cmd.append("--universe-only")
        if args.adjustments_limit is not None:
            cmd += ["--limit-symbols", str(args.adjustments_limit)]
        _run(cmd, optional=True)

    print("\nAll requested downloads finished (optional steps may have been skipped).")
    print("Next: run Phase-2 pipeline (see scripts/README.md / PLAN.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
