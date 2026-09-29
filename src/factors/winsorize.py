"""Cross-sectional winsorization and z-scoring.

Winsorize signals at the 1st/99th percentiles (or 3 MADs) within each date
before regression or z-scoring, per quant-math rules.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

WinsorMethod = Literal["percentile", "mad"]


def winsorize_cross_section(
    df: pl.DataFrame,
    column: str,
    *,
    date_col: str = "trade_date",
    lower: float = 0.01,
    upper: float = 0.99,
    method: WinsorMethod = "percentile",
    mad_k: float = 3.0,
    out_col: str | None = None,
) -> pl.DataFrame:
    """Winsorize ``column`` within each ``date_col`` group (vectorized)."""
    dest = out_col or column
    if method == "percentile":
        q = df.group_by(date_col).agg(
            pl.col(column).quantile(lower).alias("_lo"),
            pl.col(column).quantile(upper).alias("_hi"),
        )
        joined = df.join(q, on=date_col, how="left")
        return joined.with_columns(
            pl.col(column).clip(pl.col("_lo"), pl.col("_hi")).alias(dest)
        ).drop(["_lo", "_hi"])

    if method == "mad":
        # Median Absolute Deviation around the cross-sectional median
        stats = df.group_by(date_col).agg(
            pl.col(column).median().alias("_med"),
        )
        joined = df.join(stats, on=date_col, how="left")
        joined = joined.with_columns(
            (pl.col(column) - pl.col("_med")).abs().alias("_dev")
        )
        mad = joined.group_by(date_col).agg(pl.col("_dev").median().alias("_mad"))
        joined = joined.join(mad, on=date_col, how="left")
        # Avoid zero MAD — fall back to no clipping
        lo = pl.col("_med") - pl.lit(mad_k) * pl.col("_mad")
        hi = pl.col("_med") + pl.lit(mad_k) * pl.col("_mad")
        return (
            joined.with_columns(
                pl.when(pl.col("_mad").fill_null(0) <= 0)
                .then(pl.col(column))
                .otherwise(pl.col(column).clip(lo, hi))
                .alias(dest)
            ).drop(["_med", "_dev", "_mad"])
        )

    raise ValueError(f"Unknown winsor method: {method}")


def zscore_cross_section(
    df: pl.DataFrame,
    column: str,
    *,
    date_col: str = "trade_date",
    out_col: str | None = None,
) -> pl.DataFrame:
    """Cross-sectional z-score of ``column`` within each date."""
    dest = out_col or f"{column}_z"
    stats = df.group_by(date_col).agg(
        pl.col(column).mean().alias("_mu"),
        pl.col(column).std().alias("_sd"),
    )
    joined = df.join(stats, on=date_col, how="left")
    return joined.with_columns(
        pl.when(pl.col("_sd").fill_null(0) <= 0)
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise((pl.col(column) - pl.col("_mu")) / pl.col("_sd"))
        .alias(dest)
    ).drop(["_mu", "_sd"])


def prepare_signal(
    df: pl.DataFrame,
    column: str,
    *,
    date_col: str = "trade_date",
    winsor: bool = True,
    zscore: bool = True,
    method: WinsorMethod = "percentile",
) -> pl.DataFrame:
    """Winsorize then optionally z-score a characteristic for regression inputs."""
    out = df
    work_col = column
    if winsor:
        out = winsorize_cross_section(
            out, column, date_col=date_col, method=method, out_col=f"{column}_w"
        )
        work_col = f"{column}_w"
    if zscore:
        out = zscore_cross_section(out, work_col, date_col=date_col, out_col=f"{column}_z")
    return out
