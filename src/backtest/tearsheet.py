"""Phase-5 research tearsheet: correlations, two-pass FM, plots, master report.

Consumes Phase-3 factor returns / characteristics and Phase-4 report artifacts,
then writes a consolidated Markdown report + optional PNG charts under
``data/processed/reports/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from src.backtest.fama_macbeth import FMResult, fama_macbeth_two_pass, newey_west_mean_se
from src.backtest.metrics import summarize_returns
from src.backtest.report import REPORT_DIR, export_fm_report, to_markdown_table, write_parquet
from src.data.paths import ensure_data_dirs

logger = logging.getLogger(__name__)


@dataclass
class TearsheetResult:
    """Artifacts from :func:`build_tearsheet`."""

    master_md: Path
    factor_corr: pl.DataFrame
    factor_metrics: pl.DataFrame
    two_pass: FMResult | None = None
    market: pl.DataFrame | None = None
    plot_paths: list[Path] = field(default_factory=list)
    extra_paths: dict[str, Path] = field(default_factory=dict)


def build_equal_weight_market(
    returns: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    ret_col: str = "excess_ret",
    out_col: str = "MKT",
) -> pl.DataFrame:
    """Equal-weight cross-sectional mean of excess (or raw) returns per day."""
    if ret_col not in returns.columns:
        ret_col = "ret" if "ret" in returns.columns else ret_col
    if ret_col not in returns.columns:
        raise KeyError("Need excess_ret or ret to build market factor")
    return (
        returns.filter(pl.col(ret_col).is_not_null())
        .group_by(date_col)
        .agg(pl.col(ret_col).mean().alias(out_col))
        .sort(date_col)
    )


def factor_correlation_matrix(
    factor_returns: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    cols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Pearson correlation of daily factor return columns."""
    use = list(cols) if cols is not None else [c for c in factor_returns.columns if c != date_col]
    use = [c for c in use if c in factor_returns.columns]
    if len(use) < 2:
        return pl.DataFrame({"factor": use})
    sub = factor_returns.select(use).drop_nulls()
    if sub.height < 3:
        return pl.DataFrame({"factor": use})
    corr = sub.corr()
    # Polars corr() returns a square frame without an index column in recent versions
    if "factor" not in corr.columns and corr.height == len(use):
        corr = corr.with_columns(pl.Series("factor", use)).select(["factor", *use])
    return corr


def factor_metrics_table(
    factor_returns: pl.DataFrame,
    *,
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """Sharpe / NW mean diagnostics for each factor column."""
    rows: list[dict[str, object]] = []
    for col in factor_returns.columns:
        if col == date_col:
            continue
        arr = factor_returns.get_column(col).drop_nulls().to_numpy()
        if arr.size < 2:
            continue
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


def _save_wealth_plots(
    factor_returns: pl.DataFrame,
    dest_dir: Path,
    *,
    date_col: str = "trade_date",
) -> list[Path]:
    """Cumulative wealth charts for each factor (and optional backtest series)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        logger.warning("matplotlib not available — skipping plots")
        return []

    paths: list[Path] = []
    fac_cols = [c for c in factor_returns.columns if c != date_col]
    if not fac_cols:
        return paths

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in fac_cols:
        sub = factor_returns.select([date_col, col]).drop_nulls().sort(date_col)
        if sub.height < 2:
            continue
        r = sub.get_column(col).to_numpy()
        wealth = np.cumprod(1.0 + np.where(np.isfinite(r), r, 0.0))
        ax.plot(sub.get_column(date_col).to_list(), wealth, label=col, linewidth=1.2)
    ax.set_title("Factor cumulative wealth (gross, no TC)")
    ax.set_ylabel("Wealth (start=1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = dest_dir / "factor_wealth.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    paths.append(out)
    return paths


def _read_md_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def build_master_markdown(
    *,
    factor_metrics: pl.DataFrame,
    factor_corr: pl.DataFrame,
    two_pass: FMResult | None,
    report_dir: Path,
    plot_paths: Sequence[Path],
    notes: Sequence[str] | None = None,
) -> str:
    """Assemble a single Phase-5 master report body."""
    parts: list[str] = [
        "# Phase-5 Research Tearsheet\n",
        "Microstructure-augmented factor model — Nifty 500 / NSE.\n",
        "India-local $R_f$ (RBI 91-Day T-Bill). Factors use adjusted closes only.\n",
    ]
    if notes:
        parts.append("\n## Notes\n")
        for n in notes:
            parts.append(f"- {n}\n")

    parts.append("\n## Factor return metrics (Newey–West)\n\n")
    parts.append(to_markdown_table(factor_metrics) if not factor_metrics.is_empty() else "_empty_\n")

    parts.append("\n\n## Factor correlations\n\n")
    parts.append(to_markdown_table(factor_corr) if not factor_corr.is_empty() else "_empty_\n")

    # Embed Phase-4 FM if present
    fm_md = _read_md_if_exists(report_dir / "fama_macbeth_lambdas.md")
    if fm_md:
        parts.append("\n\n## Characteristic Fama–MacBeth (from Phase-4)\n\n")
        # strip leading H1 to avoid duplicate titles
        body = fm_md.split("\n", 2)[-1] if fm_md.startswith("#") else fm_md
        parts.append(body)

    if two_pass is not None and not two_pass.lambdas.is_empty():
        parts.append("\n\n## Classical two-pass Fama–MacBeth (betas → premia)\n\n")
        parts.append(to_markdown_table(two_pass.lambdas))
        parts.append(f"\n\n_NW lags = {two_pass.nw_lags}_\n")

    for name in ("bt_illiq_summary.md", "bt_momentum_summary.md", "factor_metrics.md"):
        md = _read_md_if_exists(report_dir / name)
        if md:
            parts.append(f"\n\n## Embedded: `{name}`\n\n")
            parts.append(md if not md.startswith("#") else "\n".join(md.split("\n")[1:]))

    if plot_paths:
        parts.append("\n\n## Charts\n\n")
        for p in plot_paths:
            parts.append(f"![{p.stem}]({p.name})\n\n")

    parts.append(
        "\n---\n"
        "_Generated by `scripts/run_phase5.py`. "
        "SMB/HML appear only when PIT fundamentals are available._\n"
    )
    return "".join(parts)


def build_tearsheet(
    *,
    factor_returns: pl.DataFrame,
    stock_returns: pl.DataFrame | None = None,
    dest_dir: Path | None = None,
    run_two_pass: bool = True,
    make_plots: bool = True,
    min_ts_obs: int = 60,
) -> TearsheetResult:
    """End-to-end Phase-5 tearsheet from factor (+ optional stock) returns."""
    ensure_data_dirs()
    root = dest_dir or REPORT_DIR
    root.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    fac = factor_returns.sort("trade_date")

    # Attach equal-weight market if stock panel provided
    market: pl.DataFrame | None = None
    if stock_returns is not None and not stock_returns.is_empty():
        market = build_equal_weight_market(stock_returns)
        fac = fac.join(market, on="trade_date", how="left")
        notes.append("MKT = equal-weight mean of stock excess returns.")
    else:
        notes.append("MKT not attached (no stock returns panel passed).")

    present = [c for c in fac.columns if c != "trade_date"]
    if "SMB" not in present:
        notes.append("SMB skipped — no non-null size/ME in Phase-3.")
    if "HML" not in present:
        notes.append("HML skipped — no non-null book-to-market in Phase-3.")

    metrics = factor_metrics_table(fac)
    corr = factor_correlation_matrix(fac)
    write_parquet(metrics, root / "phase5_factor_metrics.parquet")
    write_parquet(corr, root / "phase5_factor_corr.parquet")
    if market is not None:
        write_parquet(market, root / "phase5_market.parquet")

    two_pass: FMResult | None = None
    extra: dict[str, Path] = {
        "factor_metrics": root / "phase5_factor_metrics.parquet",
        "factor_corr": root / "phase5_factor_corr.parquet",
    }
    if run_two_pass and stock_returns is not None and not stock_returns.is_empty():
        factor_cols = [c for c in ("MKT", "WML", "ILLIQ", "SMB", "HML") if c in fac.columns]
        if len(factor_cols) >= 1:
            logger.info("Running two-pass FM on factors: %s", factor_cols)
            # Two-pass on full panel can be heavy; use liquid days with non-null ret
            panel = stock_returns.filter(pl.col("ret").is_not_null())
            if "excess_ret" not in panel.columns and "ret" in panel.columns:
                panel = panel.with_columns(pl.col("ret").alias("excess_ret"))
            two_pass = fama_macbeth_two_pass(
                panel,
                fac.select(["trade_date", *factor_cols]),
                factor_cols=factor_cols,
                min_ts_obs=min_ts_obs,
            )
            paths = export_fm_report(two_pass, dest_dir=root, stem="two_pass_fm")
            extra.update({f"two_pass_{k}": v for k, v in paths.items()})
        else:
            notes.append("Two-pass FM skipped — no factor columns.")
    elif run_two_pass:
        notes.append("Two-pass FM skipped — pass stock returns to enable.")

    plot_paths: list[Path] = []
    if make_plots:
        plot_paths = _save_wealth_plots(fac, root)

    master_body = build_master_markdown(
        factor_metrics=metrics,
        factor_corr=corr,
        two_pass=two_pass,
        report_dir=root,
        plot_paths=plot_paths,
        notes=notes,
    )
    master_path = root / "MASTER_REPORT.md"
    master_path.write_text(master_body, encoding="utf-8")
    logger.info("Wrote master tearsheet %s", master_path)

    return TearsheetResult(
        master_md=master_path,
        factor_corr=corr,
        factor_metrics=metrics,
        two_pass=two_pass,
        market=market,
        plot_paths=plot_paths,
        extra_paths=extra,
    )
