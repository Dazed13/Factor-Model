"""Shared vectorized sort / long-short helpers for factor portfolios."""

from __future__ import annotations

from datetime import date
from typing import Literal

import polars as pl

Weighting = Literal["equal", "value"]


def month_end_dates(dates: pl.Series | list[date]) -> list[date]:
    """Return unique month-end trading dates (last available date per YYYY-MM)."""
    s = dates if isinstance(dates, pl.Series) else pl.Series("d", dates)
    df = pl.DataFrame({"trade_date": s.cast(pl.Date)}).unique().sort("trade_date")
    return (
        df.with_columns(
            pl.col("trade_date").dt.truncate("1mo").alias("_ym"),
        )
        .group_by("_ym")
        .agg(pl.col("trade_date").max())
        .sort("trade_date")
        .get_column("trade_date")
        .to_list()
    )


def assign_quantile_buckets(
    df: pl.DataFrame,
    column: str,
    *,
    date_col: str = "trade_date",
    n: int = 5,
    out_col: str = "bucket",
    ascending: bool = True,
) -> pl.DataFrame:
    """Assign 1..n quantile buckets within each date (1 = lowest if ascending)."""
    # rank-based qcut equivalent
    ranked = df.with_columns(
        pl.col(column)
        .rank(method="average", descending=not ascending)
        .over(date_col)
        .alias("_rank"),
        pl.col(column).is_not_null().sum().over(date_col).alias("_n"),
    )
    # bucket = ceil(rank / n * n_buckets) clipped to [1, n]
    return ranked.with_columns(
        pl.when(pl.col(column).is_null() | (pl.col("_n") < n))
        .then(pl.lit(None).cast(pl.Int32))
        .otherwise(
            ((pl.col("_rank") - 1) / pl.col("_n") * n)
            .floor()
            .clip(0, n - 1)
            .cast(pl.Int32)
            + 1
        )
        .alias(out_col)
    ).drop(["_rank", "_n"])


def assign_median_split(
    df: pl.DataFrame,
    column: str,
    *,
    date_col: str = "trade_date",
    out_col: str = "size_bucket",
    low_label: str = "S",
    high_label: str = "B",
) -> pl.DataFrame:
    """Median split: below/equal median → low_label, above → high_label."""
    med = df.group_by(date_col).agg(pl.col(column).median().alias("_med"))
    joined = df.join(med, on=date_col, how="left")
    return joined.with_columns(
        pl.when(pl.col(column).is_null())
        .then(pl.lit(None).cast(pl.Utf8))
        .when(pl.col(column) <= pl.col("_med"))
        .then(pl.lit(low_label))
        .otherwise(pl.lit(high_label))
        .alias(out_col)
    ).drop("_med")


def assign_tercile_labels(
    df: pl.DataFrame,
    column: str,
    *,
    date_col: str = "trade_date",
    out_col: str = "value_bucket",
    low_label: str = "L",
    mid_label: str = "N",
    high_label: str = "H",
    low_q: float = 0.3,
    high_q: float = 0.7,
) -> pl.DataFrame:
    """Fama–French style 30/70 breakpoints for High / Neutral / Low."""
    br = df.group_by(date_col).agg(
        pl.col(column).quantile(low_q).alias("_lo"),
        pl.col(column).quantile(high_q).alias("_hi"),
    )
    joined = df.join(br, on=date_col, how="left")
    return joined.with_columns(
        pl.when(pl.col(column).is_null())
        .then(pl.lit(None).cast(pl.Utf8))
        .when(pl.col(column) <= pl.col("_lo"))
        .then(pl.lit(low_label))
        .when(pl.col(column) >= pl.col("_hi"))
        .then(pl.lit(high_label))
        .otherwise(pl.lit(mid_label))
        .alias(out_col)
    ).drop(["_lo", "_hi"])


def long_short_returns(
    panel: pl.DataFrame,
    *,
    long_mask: pl.Expr,
    short_mask: pl.Expr,
    date_col: str = "trade_date",
    ret_col: str = "ret",
    weight_col: str | None = "market_cap",
    weighting: Weighting = "value",
    factor_name: str = "factor",
) -> pl.DataFrame:
    """Dollar-neutral long-short factor return by date.

    Long and short sleeves are each normalized to sum |w| = 0.5 so that
    ``sum(w) = 0`` and gross exposure is 1.0.
    """
    work = panel.filter(pl.col(ret_col).is_not_null()).with_columns(
        long_mask.alias("_long"),
        short_mask.alias("_short"),
    )

    if weighting == "value" and weight_col and weight_col in work.columns:
        w = pl.col(weight_col).cast(pl.Float64).fill_null(0).clip(lower_bound=0)
    else:
        w = pl.lit(1.0)

    work = work.with_columns(w.alias("_raw_w"))

    def _sleeve(mask_col: str, sign: float) -> pl.DataFrame:
        sleeve = work.filter(pl.col(mask_col))
        totals = sleeve.group_by(date_col).agg(pl.col("_raw_w").sum().alias("_tot"))
        sleeve = sleeve.join(totals, on=date_col, how="left")
        return sleeve.with_columns(
            pl.when(pl.col("_tot") <= 0)
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(sign * 0.5 * pl.col("_raw_w") / pl.col("_tot"))
            .alias("_w")
        )

    long = _sleeve("_long", +1.0)
    short = _sleeve("_short", -1.0)
    legs = pl.concat([long, short], how="diagonal_relaxed")
    return (
        legs.group_by(date_col)
        .agg(
            (pl.col("_w") * pl.col(ret_col)).sum().alias(factor_name),
            pl.col("_w").sum().alias("_w_sum"),
        )
        .with_columns(pl.col(factor_name))
        .select([date_col, factor_name])
        .sort(date_col)
    )
