#!/usr/bin/env python3
"""Run Phase-4: Fama–MacBeth premia, factor metrics, optional L/S backtests.

Reads Phase-3 outputs under ``data/processed/factors/`` and writes reports to
``data/processed/reports/``.

Examples
--------
    python scripts/run_phase4.py
    python scripts/run_phase4.py --skip-backtest
    python scripts/run_phase4.py --chars mom_12_1_z,illiq_signal_z -v
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

from src.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from src.backtest.fama_macbeth import (  # noqa: E402
    fama_macbeth_characteristics,
    newey_west_mean_se,
)
from src.backtest.metrics import metrics_frame, summarize_returns  # noqa: E402
from src.backtest.report import (  # noqa: E402
    REPORT_DIR,
    export_backtest_report,
    export_fm_report,
    to_markdown_table,
    write_parquet,
)
from src.data.paths import PROCESSED_FACTORS_DIR, ensure_data_dirs  # noqa: E402
from src.data.store import read_partitioned_parquet  # noqa: E402

# Preferred FM regressors (z-scored winsorized signals from Phase-3).
DEFAULT_CHARS = ("mom_12_1_z", "illiq_signal_z", "log_me_z", "btm_z")
FALLBACK_CHARS = ("mom_12_1", "illiq_signal", "log_me", "me", "btm", "book_to_market")


def _load_partitioned(directory: Path) -> pl.DataFrame:
    df = read_partitioned_parquet(directory)
    if not df.is_empty():
        return df
    parts = sorted(directory.glob("year=*/part.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No Parquet under {directory}. Run scripts/run_phase3.py first."
        )
    return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")


def _usable_chars(panel: pl.DataFrame, requested: list[str] | None) -> list[str]:
    """Pick characteristic columns that exist and are not entirely null."""
    if requested:
        candidates = requested
    else:
        # Prefer z-scored signals; fall back to raw only if no z-cols exist.
        z_ok = [
            c
            for c in DEFAULT_CHARS
            if c in panel.columns and panel[c].null_count() < panel.height
        ]
        if z_ok:
            return z_ok
        candidates = list(FALLBACK_CHARS)
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c in seen or c not in panel.columns:
            continue
        seen.add(c)
        if panel[c].null_count() < panel.height:
            out.append(c)
    return out


def _factor_metrics_table(factor_returns: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for col in factor_returns.columns:
        if col == "trade_date":
            continue
        series = factor_returns.get_column(col).drop_nulls()
        if series.is_empty():
            continue
        arr = series.to_numpy()
        mean, nw_se, t_stat, p_value = newey_west_mean_se(arr)
        sm = summarize_returns(arr)
        rows.append(
            {
                "factor": col,
                "n_obs": sm["n_obs"],
                "mean_daily": mean,
                "nw_se": nw_se,
                "t_stat": t_stat,
                "p_value": p_value,
                "ann_mean": sm["ann_mean"],
                "ann_vol": sm["ann_vol"],
                "sharpe": sm["sharpe"],
                "max_drawdown": sm["max_drawdown"],
                "hit_rate": sm["hit_rate"],
                "total_return": sm["total_return"],
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase-4 FM + diagnostics")
    parser.add_argument(
        "--factors-dir",
        type=Path,
        default=PROCESSED_FACTORS_DIR,
        help="Phase-3 output root (default: data/processed/factors)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="Report output directory (default: data/processed/reports)",
    )
    parser.add_argument(
        "--chars",
        type=str,
        default=None,
        help="Comma-separated FM characteristic columns (default: auto)",
    )
    parser.add_argument(
        "--nw-lags",
        type=int,
        default=None,
        help="Newey–West lags (default: automatic rule-of-thumb)",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip month-end quantile long–short backtests",
    )
    parser.add_argument(
        "--skip-fm",
        action="store_true",
        help="Skip Fama–MacBeth characteristic regressions",
    )
    parser.add_argument(
        "--tc-bps",
        type=float,
        default=0.0,
        help="One-way transaction cost in bps for backtests (default: 0)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_data_dirs()
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    char_dir = args.factors_dir / "characteristics"
    fac_dir = args.factors_dir / "factor_returns"
    panel = _load_partitioned(char_dir)
    factor_returns = _load_partitioned(fac_dir)
    logging.info(
        "Loaded characteristics rows=%d symbols=%d; factor days=%d",
        panel.height,
        panel["symbol"].n_unique() if "symbol" in panel.columns else 0,
        factor_returns.height,
    )

    requested = (
        [c.strip() for c in args.chars.split(",") if c.strip()] if args.chars else None
    )
    chars = _usable_chars(panel, requested)
    logging.info("FM characteristics: %s", chars or "(none)")

    # --- Factor time-series metrics (WML / ILLIQ / …) ---
    fac_metrics = _factor_metrics_table(factor_returns)
    if not fac_metrics.is_empty():
        write_parquet(fac_metrics, report_dir / "factor_metrics.parquet")
        md = report_dir / "factor_metrics.md"
        md.write_text(
            "# Factor return metrics (Newey–West mean SE)\n\n"
            + to_markdown_table(fac_metrics)
            + "\n",
            encoding="utf-8",
        )
        print("\n=== Factor metrics ===")
        print(to_markdown_table(fac_metrics))

    # --- Fama–MacBeth ---
    fm_paths: dict[str, Path] = {}
    if not args.skip_fm:
        if not chars:
            logging.warning("No usable characteristics for FM — skipping")
        else:
            # Prefer excess returns; drop rows missing all selected chars
            work = panel.filter(pl.col("ret").is_not_null())
            fm = fama_macbeth_characteristics(
                work,
                chars,
                ret_col="excess_ret" if "excess_ret" in work.columns else "ret",
                nw_lags=args.nw_lags,
            )
            fm_paths = export_fm_report(fm, dest_dir=report_dir, stem="fama_macbeth")
            print("\n=== Fama–MacBeth premia ===")
            if fm.lambdas.is_empty():
                print("(empty — insufficient cross-sections)")
            else:
                print(to_markdown_table(fm.lambdas))
                print(f"_NW lags = {fm.nw_lags}_")

    # --- Optional characteristic backtests ---
    bt_summaries: list[pl.DataFrame] = []
    if not args.skip_backtest:
        # Map FM-friendly z-cols back to raw signals for sorts
        signal_specs = [
            ("mom_12_1", True, "bt_momentum"),
            ("illiq_signal", True, "bt_illiq"),
        ]
        for signal, long_high, stem in signal_specs:
            if signal not in panel.columns or panel[signal].null_count() == panel.height:
                logging.info("Skipping backtest for missing signal %s", signal)
                continue
            use_value = (
                "market_cap" in panel.columns
                and panel["market_cap"].null_count() < panel.height
            )
            cfg = BacktestConfig(
                signal_col=signal,
                kind="quintile",
                long_high=long_high,
                weighting="value" if use_value else "equal",
                weight_col="market_cap" if use_value else None,
                frequency="month_end",
                ret_col="ret",
                # Engine expects proportional cost (0.001 = 10 bps); CLI is in bps.
                tc_bps=(args.tc_bps / 10_000.0) if args.tc_bps else 0.0,
            )
            logging.info(
                "Running backtest signal=%s weighting=%s",
                signal,
                cfg.weighting,
            )
            result = run_backtest(panel, cfg)
            if result.returns.is_empty():
                logging.warning("Empty backtest returns for %s", signal)
                continue
            paths = export_backtest_report(
                result.returns,
                weights=result.weights,
                dest_dir=report_dir,
                stem=stem,
                ret_col="net_ret",
            )
            m = summarize_returns(result.returns, ret_col="net_ret")
            mframe = metrics_frame(
                {**m, **result.metrics},
                name=stem,
            )
            bt_summaries.append(mframe)
            print(f"\n=== Backtest {stem} ({signal}) ===")
            print(to_markdown_table(mframe))
            logging.info("Wrote %s", paths.get("markdown"))

    if bt_summaries:
        combined = pl.concat(bt_summaries, how="diagonal_relaxed")
        write_parquet(combined, report_dir / "backtest_metrics.parquet")

    print(f"\nReports written under {report_dir}")
    if fac_metrics.is_empty() and not fm_paths and not bt_summaries:
        logging.error("Phase-4 produced no outputs")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
