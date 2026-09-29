"""Value (book-to-market) characteristic and HML factor.

HML uses the Fama–French 2×3 Size × Value sort when market equity is available:

    HML = 1/2 (SH + BH) - 1/2 (SL + BL)
"""

from __future__ import annotations

import polars as pl

from src.factors.buckets import (
    assign_median_split,
    assign_tercile_labels,
    long_short_returns,
)
from src.factors.winsorize import prepare_signal


def attach_value_characteristic(
    df: pl.DataFrame,
    *,
    btm_col: str = "book_to_market",
    out_col: str = "btm",
) -> pl.DataFrame:
    """Attach book-to-market characteristic."""
    if btm_col not in df.columns and "btm" in df.columns:
        return df.with_columns(pl.col("btm").cast(pl.Float64).alias(out_col))
    if btm_col not in df.columns:
        raise KeyError(f"Value requires '{btm_col}' or 'btm' column")
    return df.with_columns(pl.col(btm_col).cast(pl.Float64).alias(out_col))


def construct_hml(
    panel: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    ret_col: str = "ret",
    me_col: str = "me",
    btm_col: str = "btm",
    weight_col: str = "market_cap",
) -> pl.DataFrame:
    """Build daily HML factor returns (High BTM minus Low BTM)."""
    work = panel
    if btm_col not in work.columns:
        work = attach_value_characteristic(work, out_col=btm_col)
    if me_col not in work.columns and "market_cap" in work.columns:
        work = work.with_columns(pl.col("market_cap").alias(me_col))

    work = assign_tercile_labels(work, btm_col, date_col=date_col, out_col="value_bucket")

    if me_col in work.columns:
        work = assign_median_split(work, me_col, date_col=date_col, out_col="size_bucket")
        # HML = 0.5*(SH+BH) - 0.5*(SL+BL)  → long H within both sizes, short L
        return long_short_returns(
            work,
            long_mask=(pl.col("value_bucket") == "H")
            & pl.col("size_bucket").is_in(["S", "B"]),
            short_mask=(pl.col("value_bucket") == "L")
            & pl.col("size_bucket").is_in(["S", "B"]),
            date_col=date_col,
            ret_col=ret_col,
            weight_col=weight_col if weight_col in work.columns else me_col,
            factor_name="HML",
        )

    return long_short_returns(
        work,
        long_mask=pl.col("value_bucket") == "H",
        short_mask=pl.col("value_bucket") == "L",
        date_col=date_col,
        ret_col=ret_col,
        weight_col=weight_col if weight_col in work.columns else None,
        weighting="value" if weight_col in work.columns else "equal",
        factor_name="HML",
    )


def value_signal_panel(
    df: pl.DataFrame,
    *,
    winsor: bool = True,
) -> pl.DataFrame:
    """BTM characteristic prepared for cross-sectional regressions."""
    out = attach_value_characteristic(df)
    return prepare_signal(out, "btm", winsor=winsor)
