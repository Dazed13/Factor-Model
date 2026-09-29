"""Export factor diagnostics to Parquet and Markdown-friendly tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import polars as pl

from src.backtest.fama_macbeth import FMResult
from src.backtest.metrics import metrics_frame, quantile_metrics
from src.data.paths import PROCESSED_DIR, ensure_data_dirs

logger = logging.getLogger(__name__)

REPORT_DIR: Path = PROCESSED_DIR / "reports"


def _ensure_report_dir(dest: Path | None = None) -> Path:
    ensure_data_dirs()
    path = dest or REPORT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def fm_result_to_frame(result: FMResult) -> pl.DataFrame:
    """Lambda summary with NW lags annotated."""
    if result.lambdas.is_empty():
        return result.lambdas
    return result.lambdas.with_columns(pl.lit(result.nw_lags).alias("nw_lags"))


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="snappy")
    return path


def to_markdown_table(df: pl.DataFrame, *, float_precision: int = 4) -> str:
    """Render a small Polars frame as a GitHub-flavored Markdown table."""
    if df.is_empty():
        return "_empty_"

    cols = df.columns

    def _fmt(val: object) -> str:
        if val is None:
            return ""
        if isinstance(val, float):
            if val != val:  # NaN
                return ""
            return f"{val:.{float_precision}f}"
        return str(val)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def export_fm_report(
    result: FMResult,
    *,
    dest_dir: Path | None = None,
    stem: str = "fama_macbeth",
) -> dict[str, Path]:
    """Write FM lambda table + gamma path as snappy Parquet (+ Markdown summary)."""
    root = _ensure_report_dir(dest_dir)
    lambdas = fm_result_to_frame(result)
    paths = {
        "lambdas": write_parquet(lambdas, root / f"{stem}_lambdas.parquet"),
        "gamma_path": write_parquet(
            result.gamma_path, root / f"{stem}_gamma_path.parquet"
        ),
    }
    md = root / f"{stem}_lambdas.md"
    md.write_text(
        "# Fama–MacBeth Premia (Newey–West SEs)\n\n"
        + to_markdown_table(lambdas)
        + f"\n\n_NW lags = {result.nw_lags}_\n",
        encoding="utf-8",
    )
    paths["markdown"] = md
    logger.info("Wrote FM report to %s", root)
    return paths


def export_backtest_report(
    returns: pl.DataFrame,
    *,
    weights: pl.DataFrame | None = None,
    quantile_returns: pl.DataFrame | None = None,
    dest_dir: Path | None = None,
    stem: str = "backtest",
    ret_col: str = "net_ret",
) -> dict[str, Path]:
    """Persist portfolio metrics and optional per-quantile diagnostics."""
    root = _ensure_report_dir(dest_dir)
    paths: dict[str, Path] = {}

    from src.backtest.metrics import portfolio_metrics_bundle

    metrics = portfolio_metrics_bundle(returns, weights, ret_col=ret_col)
    mframe = metrics_frame(metrics, name=stem)
    paths["metrics"] = write_parquet(mframe, root / f"{stem}_metrics.parquet")
    paths["returns"] = write_parquet(returns, root / f"{stem}_returns.parquet")

    md_parts = [
        f"# Backtest report: `{stem}`\n",
        to_markdown_table(mframe),
    ]

    if quantile_returns is not None and not quantile_returns.is_empty():
        qmet = quantile_metrics(quantile_returns)
        paths["quantile_metrics"] = write_parquet(
            qmet, root / f"{stem}_quantile_metrics.parquet"
        )
        md_parts.extend(["\n## Quantile metrics\n", to_markdown_table(qmet)])

    md_path = root / f"{stem}_summary.md"
    md_path.write_text("\n".join(md_parts) + "\n", encoding="utf-8")
    paths["markdown"] = md_path
    return paths


def combine_diagnostics(
    *,
    fm: FMResult | None = None,
    portfolio_metrics: Mapping[str, float] | None = None,
    quantile_metrics_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Stack FM premia and portfolio metrics into one tidy diagnostic frame."""
    frames: list[pl.DataFrame] = []
    if fm is not None and not fm.lambdas.is_empty():
        frames.append(
            fm.lambdas.select(
                pl.lit("fm_premium").alias("section"),
                pl.col("name"),
                pl.col("mean").alias("value"),
                pl.col("t_stat"),
                pl.col("nw_se"),
            )
        )
    if portfolio_metrics:
        frames.append(
            pl.DataFrame(
                {
                    "section": ["portfolio"] * len(portfolio_metrics),
                    "name": list(portfolio_metrics.keys()),
                    "value": list(portfolio_metrics.values()),
                    "t_stat": [None] * len(portfolio_metrics),
                    "nw_se": [None] * len(portfolio_metrics),
                }
            )
        )
    if quantile_metrics_df is not None and not quantile_metrics_df.is_empty():
        q = quantile_metrics_df.with_columns(
            pl.lit("quantile").alias("section"),
            pl.format("Q{}", pl.col("quantile")).alias("name"),
            pl.col("sharpe").alias("value"),
            pl.lit(None).cast(pl.Float64).alias("t_stat"),
            pl.lit(None).cast(pl.Float64).alias("nw_se"),
        ).select(["section", "name", "value", "t_stat", "nw_se"])
        frames.append(q)

    if not frames:
        return pl.DataFrame(
            schema={
                "section": pl.Utf8,
                "name": pl.Utf8,
                "value": pl.Float64,
                "t_stat": pl.Float64,
                "nw_se": pl.Float64,
            }
        )
    return pl.concat(frames, how="diagonal_relaxed")
