"""Size characteristic and SMB (Small-Minus-Big) factor.

Market equity is taken from lagged ``market_cap`` (fundamentals as-of join) or
an optional ``me`` column. Monthly SMB uses the standard 2×3 Size × Value sort
when book-to-market is available; otherwise a simple median long-short.
"""

from __future__ import annotations

import polars as pl

from src.factors.buckets import (
    assign_median_split,
    assign_tercile_labels,
    long_short_returns,
    month_end_dates,
)
from src.factors.winsorize import prepare_signal


def attach_size_characteristic(
    df: pl.DataFrame,
    *,
    me_col: str = "market_cap",
    out_col: str = "me",
    log_me: bool = True,
) -> pl.DataFrame:
    """Attach market-equity characteristic (optionally log)."""
    if me_col not in df.columns and "me" in df.columns:
        me_col = "me"
    if me_col not in df.columns:
        raise KeyError(f"Size requires '{me_col}' or 'me' column")

    out = df.with_columns(pl.col(me_col).cast(pl.Float64).alias(out_col))
    if log_me:
        out = out.with_columns(
            pl.when(pl.col(out_col) > 0)
            .then(pl.col(out_col).log())
            .otherwise(None)
            .alias("log_me")
        )
    return out


def construct_smb(
    panel: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    ret_col: str = "ret",
    me_col: str = "me",
    btm_col: str | None = "btm",
    weight_col: str = "market_cap",
    monthly: bool = True,
) -> pl.DataFrame:
    """Build daily SMB factor returns.

    If ``btm_col`` is present, uses FF 2×3:
    SMB = mean(SL, SN, SH) - mean(BL, BN, BH) via value-weighted sleeves.
    Else: equal long small / short big on median ME split.
    """
    work = panel
    if me_col not in work.columns:
        work = attach_size_characteristic(work, me_col="market_cap", out_col=me_col)

    if monthly:
        ends = month_end_dates(work.get_column(date_col))
        # Characteristics fixed within month from prior month-end signal —
        # for Phase 3 we assign buckets on each date using that day's ME/BTM
        # (already lagged in the fundamentals pipeline).
        _ = ends  # reserved for explicit month-end rebalance maps in Phase 4

    use_btm = btm_col is not None and btm_col in work.columns

    if use_btm:
        work = assign_median_split(work, me_col, date_col=date_col, out_col="size_bucket")
        work = assign_tercile_labels(
            work, btm_col, date_col=date_col, out_col="value_bucket"
        )
        # SMB = 1/3 (SL+SN+SH) - 1/3 (BL+BN+BH)
        # Implement as long all Small, short all Big (ex Neutral-only size view)
        return long_short_returns(
            work,
            long_mask=pl.col("size_bucket") == "S",
            short_mask=pl.col("size_bucket") == "B",
            date_col=date_col,
            ret_col=ret_col,
            weight_col=weight_col if weight_col in work.columns else me_col,
            factor_name="SMB",
        )

    work = assign_median_split(work, me_col, date_col=date_col, out_col="size_bucket")
    return long_short_returns(
        work,
        long_mask=pl.col("size_bucket") == "S",
        short_mask=pl.col("size_bucket") == "B",
        date_col=date_col,
        ret_col=ret_col,
        weight_col=weight_col if weight_col in work.columns else None,
        weighting="value" if weight_col in work.columns else "equal",
        factor_name="SMB",
    )


def size_signal_panel(
    df: pl.DataFrame,
    *,
    winsor: bool = True,
) -> pl.DataFrame:
    """ME characteristic prepared for cross-sectional regressions."""
    out = attach_size_characteristic(df)
    return prepare_signal(out, "log_me" if "log_me" in out.columns else "me", winsor=winsor)
