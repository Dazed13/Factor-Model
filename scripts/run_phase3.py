#!/usr/bin/env python3
"""Run Phase-3: characteristic panel + factor returns (WML, ILLIQ; SMB/HML if ME/BTM).

Reads ``data/processed/returns/``, writes under ``data/processed/factors/``.

Examples
--------
    python scripts/run_phase3.py
    python scripts/run_phase3.py --amihud-window 30
    python scripts/run_phase3.py --no-winsorize
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

from src.data.paths import PROCESSED_FACTORS_DIR, PROCESSED_RETURNS_DIR, ensure_data_dirs  # noqa: E402
from src.data.store import read_partitioned_parquet  # noqa: E402
from src.factors.amihud import DEFAULT_WINDOW  # noqa: E402
from src.factors.combine import build_factors  # noqa: E402


def _load_returns(directory: Path) -> pl.DataFrame:
    df = read_partitioned_parquet(directory)
    if df.is_empty():
        # fallback: year=*/part.parquet explicit
        parts = sorted(directory.glob("year=*/part.parquet"))
        if not parts:
            raise FileNotFoundError(
                f"No returns Parquet under {directory}. Run scripts/run_phase2.py first."
            )
        df = pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")
    return df


def _summarize_factor(df: pl.DataFrame, col: str) -> str:
    if col not in df.columns:
        return f"{col}: (missing)"
    s = df[col].drop_nulls()
    if s.is_empty():
        return f"{col}: all null"
    return (
        f"{col}: n={s.len()} mean={float(s.mean()):.6f} "
        f"std={float(s.std()):.6f} "
        f"[{float(s.min()):.4f}, {float(s.max()):.4f}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase-3 factor construction")
    parser.add_argument(
        "--returns-dir",
        type=Path,
        default=PROCESSED_RETURNS_DIR,
        help="Processed returns root (default: data/processed/returns)",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=PROCESSED_FACTORS_DIR,
        help="Output root (default: data/processed/factors)",
    )
    parser.add_argument(
        "--amihud-window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Amihud rolling window in trading days (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--no-winsorize",
        action="store_true",
        help="Skip cross-sectional winsorization / z-scores on signals",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Compute in-memory only (do not write Parquet)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_data_dirs()

    returns = _load_returns(args.returns_dir)
    logging.info(
        "Loaded returns rows=%d symbols=%d dates=%s→%s",
        returns.height,
        returns["symbol"].n_unique() if "symbol" in returns.columns else 0,
        returns["trade_date"].min() if returns.height else None,
        returns["trade_date"].max() if returns.height else None,
    )

    # Filter liquidated / suspended rows with null returns from portfolio sorts
    if "ret" in returns.columns:
        before = returns.height
        returns = returns.filter(pl.col("ret").is_not_null())
        logging.info("Dropped null-ret rows: %d → %d", before, returns.height)

    result = build_factors(
        returns,
        amihud_window=args.amihud_window,
        winsorize=not args.no_winsorize,
        dest_dir=args.dest_dir,
        persist=not args.no_persist,
    )

    chars = result.characteristics
    factors = result.factor_returns

    print("\n=== Phase-3 complete ===")
    print(f"characteristics rows : {chars.height:,}")
    if chars.height:
        signal_cols = [
            c
            for c in (
                "illiq",
                "illiq_roll",
                "illiq_signal",
                "mom_12_1",
                "me",
                "log_me",
                "btm",
            )
            if c in chars.columns
        ]
        for c in signal_cols:
            nn = chars.height - chars[c].null_count()
            print(f"  {c:<16} non-null={nn:,} / {chars.height:,}")

    print(f"factor returns rows  : {factors.height:,}")
    if factors.height:
        print(
            f"factor date range    : {factors['trade_date'].min()} → {factors['trade_date'].max()}"
        )
        for col in factors.columns:
            if col == "trade_date":
                continue
            print(f"  {_summarize_factor(factors, col)}")

    if result.characteristic_paths:
        print(f"\ncharacteristics → {args.dest_dir / 'characteristics'}")
        print(f"  partitions: {len(result.characteristic_paths)}")
    if result.factor_paths:
        print(f"factor returns  → {args.dest_dir / 'factor_returns'}")
        print(f"  partitions: {len(result.factor_paths)}")

    fac_names = [c for c in factors.columns if c != "trade_date"]
    if not fac_names:
        logging.error("No factor columns produced")
        return 1
    if "WML" not in fac_names and "ILLIQ" not in fac_names:
        logging.warning("Neither WML nor ILLIQ produced — check returns / volume columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
