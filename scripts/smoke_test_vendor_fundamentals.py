#!/usr/bin/env python3
"""Smoke-test EODHD and FMP fundamentals APIs.

Reads keys from the environment (never hard-coded):

* ``EODHD_API_TOKEN`` (or ``EODHD_API_KEY``) — free signup at eodhd.com
  Demo token ``demo`` works for a fixed US set (AAPL.US, …) without signup.
* ``FMP_API_KEY`` (or ``FMP_API_TOKEN``) — signup at financialmodelingprep.com
  (no public demo key for India; NSE tests skip if unset).

Examples
--------
    # EODHD schema check with public demo + optional NSE if token set
    python scripts/smoke_test_vendor_fundamentals.py

    EODHD_API_TOKEN=your_token FMP_API_KEY=your_key \\
      python scripts/smoke_test_vendor_fundamentals.py --symbols RELIANCE,TCS,INFY
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import ensure_repo_on_path

ensure_repo_on_path()

from src.data.fundamentals_vendors import (  # noqa: E402
    fetch_eodhd_fundamentals_quarterly,
    fetch_eodhd_fundamentals_snapshot,
    fetch_fmp_fundamentals_quarterly,
    fetch_fmp_fundamentals_snapshot,
)
from src.data.paths import PROCESSED_FUNDAMENTALS_DIR, ensure_data_dirs  # noqa: E402

logger = logging.getLogger(__name__)


def _eodhd_token() -> str | None:
    return os.environ.get("EODHD_API_TOKEN") or os.environ.get("EODHD_API_KEY")


def _fmp_key() -> str | None:
    return os.environ.get("FMP_API_KEY") or os.environ.get("FMP_API_TOKEN")


def _print_df(label: str, df) -> bool:
    ok = df is not None and getattr(df, "height", 0) > 0
    status = "OK" if ok else "EMPTY"
    print(f"\n=== {label} [{status}] ===")
    if ok:
        print(df)
    else:
        print("(no rows)")
    return bool(ok)


def run_eodhd(symbols: list[str], *, include_demo: bool) -> dict[str, bool]:
    results: dict[str, bool] = {}
    token = _eodhd_token()

    if include_demo:
        print("\n--- EODHD demo (AAPL.US, no signup) ---")
        try:
            snap = fetch_eodhd_fundamentals_snapshot(["AAPL.US"], api_token="demo")
            results["eodhd_demo_snapshot"] = _print_df("EODHD demo snapshot", snap)
            q = fetch_eodhd_fundamentals_quarterly(
                ["AAPL.US"], api_token="demo", max_periods=4
            )
            results["eodhd_demo_quarterly"] = _print_df("EODHD demo quarterly", q)
        except Exception as exc:  # noqa: BLE001
            logger.exception("EODHD demo failed: %s", exc)
            results["eodhd_demo_snapshot"] = False
            results["eodhd_demo_quarterly"] = False

    if not token:
        print(
            "\n[skip] NSE EODHD tests — set EODHD_API_TOKEN "
            "(https://eodhd.com/register) to probe RELIANCE.NSE etc."
        )
        results["eodhd_nse_snapshot"] = False
        results["eodhd_nse_quarterly"] = False
        return results

    print(f"\n--- EODHD NSE ({', '.join(symbols)}) ---")
    try:
        snap = fetch_eodhd_fundamentals_snapshot(symbols, api_token=token)
        results["eodhd_nse_snapshot"] = _print_df("EODHD NSE snapshot", snap)
        q = fetch_eodhd_fundamentals_quarterly(
            symbols, api_token=token, max_periods=4
        )
        results["eodhd_nse_quarterly"] = _print_df("EODHD NSE quarterly", q)
        if results["eodhd_nse_snapshot"] or results["eodhd_nse_quarterly"]:
            ensure_data_dirs()
            out = PROCESSED_FUNDAMENTALS_DIR / "smoke_eodhd.parquet"
            (snap if results["eodhd_nse_snapshot"] else q).write_parquet(
                out, compression="snappy"
            )
            print(f"Wrote {out}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("EODHD NSE failed: %s", exc)
        results["eodhd_nse_snapshot"] = False
        results["eodhd_nse_quarterly"] = False
    return results


def run_fmp(symbols: list[str]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    key = _fmp_key()
    if not key:
        print(
            "\n[skip] FMP tests — set FMP_API_KEY "
            "(https://site.financialmodelingprep.com/register) "
            "(no public demo key for NSE)."
        )
        results["fmp_snapshot"] = False
        results["fmp_quarterly"] = False
        return results

    # Prefer US symbol first if user only wants connectivity; then NSE list.
    probe = list(symbols)
    print(f"\n--- FMP ({', '.join(probe)}) ---")
    try:
        snap = fetch_fmp_fundamentals_snapshot(probe, api_key=key)
        results["fmp_snapshot"] = _print_df("FMP snapshot", snap)
        q = fetch_fmp_fundamentals_quarterly(probe, api_key=key, max_periods=4)
        results["fmp_quarterly"] = _print_df("FMP quarterly", q)
        if results["fmp_snapshot"] or results["fmp_quarterly"]:
            ensure_data_dirs()
            out = PROCESSED_FUNDAMENTALS_DIR / "smoke_fmp.parquet"
            (snap if results["fmp_snapshot"] else q).write_parquet(
                out, compression="snappy"
            )
            print(f"Wrote {out}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("FMP failed: %s", exc)
        results["fmp_snapshot"] = False
        results["fmp_quarterly"] = False
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test EODHD / FMP fundamentals")
    parser.add_argument(
        "--symbols",
        default="RELIANCE,TCS,INFY",
        help="Comma-separated bare NSE symbols (default: RELIANCE,TCS,INFY)",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip EODHD demo AAPL.US connectivity check",
    )
    parser.add_argument(
        "--source",
        choices=["both", "eodhd", "fmp"],
        default="both",
        help="Which vendor(s) to probe (default: both)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    results: dict[str, bool] = {}

    if args.source in {"both", "eodhd"}:
        results.update(run_eodhd(symbols, include_demo=not args.skip_demo))
    if args.source in {"both", "fmp"}:
        results.update(run_fmp(symbols))

    print("\n========== SUMMARY ==========")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL/SKIP':<10} {name}")

    # Success if at least one live probe returned rows (demo counts).
    if any(results.values()):
        print("\nSmoke test: at least one vendor path returned data.")
        if not _eodhd_token():
            print("Tip: export EODHD_API_TOKEN=... to test NSE (.NSE) coverage.")
        if not _fmp_key():
            print("Tip: export FMP_API_KEY=... to test FMP NSE (.NS) coverage.")
        return 0

    print("\nSmoke test: no vendor returned rows. Check keys / network / SSL.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
