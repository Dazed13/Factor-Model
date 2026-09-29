#!/usr/bin/env python3
"""Run Phase-2: clean prices → returns (excess vs Rf) → fundamentals as-of join.

Prefers the **processed adjusted** Bhavcopy panel written by
``download_adjustments.py``. Falls back to raw zips only with ``--from-raw``
(then uses ``adj_factor=1`` if no adjustment columns are present).

Examples
--------
    python scripts/run_phase2.py
    python scripts/run_phase2.py --from-processed
    python scripts/run_phase2.py --from-raw
    python scripts/run_phase2.py --fundamentals-csv data/raw/fundamentals/fundamentals.csv
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

from src.data.pipeline import PipelineConfig, run_phase2_pipeline  # noqa: E402
from src.data.paths import (  # noqa: E402
    PROCESSED_BHAVCOPY_DIR,
    PROCESSED_RETURNS_DIR,
    RAW_BHAVCOPY_DIR,
    ensure_data_dirs,
)


def _load_processed_bhav(directory: Path) -> pl.DataFrame:
    parts = sorted(directory.glob("year=*/part.parquet"))
    if not parts:
        # flat parquet fallback
        parts = sorted(directory.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No processed Bhavcopy Parquet under {directory}. "
            "Run download_adjustments.py first, or pass --from-raw."
        )
    logging.info("Loading %d processed Bhavcopy partition(s) from %s", len(parts), directory)
    return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase-2 clean + returns pipeline")
    parser.add_argument(
        "--from-processed",
        action="store_true",
        default=True,
        help="Load adjusted panel from data/processed/bhavcopy (default)",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Parse data/raw/bhavcopy zips instead of processed Parquet",
    )
    parser.add_argument(
        "--processed-bhavcopy-dir",
        type=Path,
        default=PROCESSED_BHAVCOPY_DIR,
        help="Processed Bhavcopy root (year=*/part.parquet)",
    )
    parser.add_argument(
        "--raw-bhavcopy-dir",
        type=Path,
        default=RAW_BHAVCOPY_DIR,
    )
    parser.add_argument(
        "--fundamentals-csv",
        type=Path,
        default=None,
        help="Optional fundamentals CSV override",
    )
    parser.add_argument(
        "--risk-free-csv",
        type=Path,
        default=None,
        help="Optional risk-free CSV override",
    )
    parser.add_argument(
        "--lag-months",
        type=int,
        default=3,
        help="Fundamentals reporting lag in months (default: 3)",
    )
    parser.add_argument(
        "--no-duckdb",
        action="store_true",
        help="Skip DuckDB view registration",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_data_dirs()

    cfg = PipelineConfig(
        raw_bhavcopy_dir=args.raw_bhavcopy_dir,
        processed_bhavcopy_dir=args.processed_bhavcopy_dir,
        fundamentals_csv=args.fundamentals_csv,
        risk_free_csv=args.risk_free_csv,
        fundamental_lag_months=args.lag_months,
    )

    bhav: pl.DataFrame | None = None
    if args.from_raw:
        logging.info("Phase-2 will load raw Bhavcopy from %s", args.raw_bhavcopy_dir)
        bhav = None
    else:
        bhav = _load_processed_bhav(args.processed_bhavcopy_dir)
        logging.info(
            "Loaded panel rows=%d cols=%d adj_close=%s",
            bhav.height,
            len(bhav.columns),
            "adj_close" in bhav.columns,
        )
        if "adj_close" not in bhav.columns and "adj_factor" not in bhav.columns:
            logging.warning(
                "Processed panel has no adj_close/adj_factor — returns will use "
                "default adj_factor=1.0. Prefer re-running download_adjustments.py."
            )

    result = run_phase2_pipeline(
        cfg,
        bhavcopy=bhav,
        register_duckdb=not args.no_duckdb,
    )

    rets = result.returns
    print("\n=== Phase-2 complete ===")
    print(f"clean prices rows : {result.clean_prices.height:,}")
    print(f"returns rows      : {rets.height:,}")
    if not rets.is_empty():
        print(f"date range        : {rets['trade_date'].min()} → {rets['trade_date'].max()}")
        print(f"symbols           : {rets['symbol'].n_unique()}")
        print(f"columns           : {rets.columns}")
        for col in ("ret", "log_ret", "excess_ret", "market_cap", "book_to_market"):
            if col in rets.columns:
                nn = rets.height - rets[col].null_count()
                print(f"  {col:<16} non-null={nn:,} / {rets.height:,}")
    print(f"returns paths     : {result.returns_paths}")
    print(f"bhavcopy paths    : {result.bhavcopy_paths}")
    if result.duckdb_views:
        print(f"duckdb views      : {result.duckdb_views}")
    print(f"\nOutputs under {PROCESSED_RETURNS_DIR}")

    if result.fundamentals is not None and not result.fundamentals.is_empty():
        avail_min = result.fundamentals["available_date"].min()
        if "trade_date" in rets.columns and rets.height and avail_min is not None:
            tmax = rets["trade_date"].max()
            if avail_min > tmax:
                logging.warning(
                    "Fundamentals available_date min=%s is after returns end=%s — "
                    "snapshot ME/BTM will not join historically. Use a PIT CSV for SMB/HML.",
                    avail_min,
                    tmax,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
