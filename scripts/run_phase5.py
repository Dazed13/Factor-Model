#!/usr/bin/env python3
"""Run Phase-5: research tearsheet (correlations, two-pass FM, master report).

Consumes Phase-3 factor returns and Phase-2/3 stock returns; embeds Phase-4
Markdown reports when present.

Examples
--------
    python scripts/run_phase5.py
    python scripts/run_phase5.py --skip-two-pass
    python scripts/run_phase5.py --no-plots
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

from src.backtest.report import REPORT_DIR, to_markdown_table  # noqa: E402
from src.backtest.tearsheet import build_tearsheet  # noqa: E402
from src.data.paths import (  # noqa: E402
    PROCESSED_FACTORS_DIR,
    PROCESSED_RETURNS_DIR,
    ensure_data_dirs,
)
from src.data.store import read_partitioned_parquet  # noqa: E402


def _load_partitioned(directory: Path, *, label: str) -> pl.DataFrame:
    df = read_partitioned_parquet(directory)
    if not df.is_empty():
        return df
    parts = sorted(directory.glob("year=*/part.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No {label} Parquet under {directory}. "
            "Run earlier phase scripts first (phase2/phase3)."
        )
    return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase-5 research tearsheet")
    parser.add_argument(
        "--factors-dir",
        type=Path,
        default=PROCESSED_FACTORS_DIR,
        help="Phase-3 factors root (default: data/processed/factors)",
    )
    parser.add_argument(
        "--returns-dir",
        type=Path,
        default=PROCESSED_RETURNS_DIR,
        help="Stock returns panel for MKT + two-pass FM",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="Report output directory",
    )
    parser.add_argument(
        "--skip-two-pass",
        action="store_true",
        help="Skip classical two-pass Fama–MacBeth (faster)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib wealth charts",
    )
    parser.add_argument(
        "--min-ts-obs",
        type=int,
        default=60,
        help="Min time-series obs per name for two-pass betas (default: 60)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_data_dirs()

    fac = _load_partitioned(args.factors_dir / "factor_returns", label="factor returns")
    logging.info(
        "Factor returns rows=%d cols=%s",
        fac.height,
        [c for c in fac.columns if c != "trade_date"],
    )

    stock: pl.DataFrame | None = None
    try:
        stock = _load_partitioned(args.returns_dir, label="stock returns")
        # Keep columns needed for MKT / two-pass to limit memory
        keep = [
            c
            for c in ("trade_date", "symbol", "ret", "excess_ret")
            if c in stock.columns
        ]
        stock = stock.select(keep).filter(pl.col("ret").is_not_null())
        logging.info("Stock returns rows=%d symbols=%d", stock.height, stock["symbol"].n_unique())
    except FileNotFoundError as exc:
        logging.warning("%s — tearsheet will omit MKT / two-pass", exc)

    result = build_tearsheet(
        factor_returns=fac,
        stock_returns=stock,
        dest_dir=args.report_dir,
        run_two_pass=not args.skip_two_pass,
        make_plots=not args.no_plots,
        min_ts_obs=args.min_ts_obs,
    )

    print("\n=== Phase-5 tearsheet ===")
    print(f"master report : {result.master_md}")
    if not result.factor_metrics.is_empty():
        print("\nFactor metrics:")
        print(to_markdown_table(result.factor_metrics))
    if not result.factor_corr.is_empty():
        print("\nFactor correlations:")
        print(to_markdown_table(result.factor_corr))
    if result.two_pass is not None and not result.two_pass.lambdas.is_empty():
        print("\nTwo-pass FM premia:")
        print(to_markdown_table(result.two_pass.lambdas))
        print(f"_NW lags = {result.two_pass.nw_lags}_")
    if result.plot_paths:
        print("\nPlots:")
        for p in result.plot_paths:
            print(f"  {p}")

    print(f"\nAll reports under {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
