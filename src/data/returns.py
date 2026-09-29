"""Daily return construction from adjusted prices + India-local excess returns.

Never compute research returns from unadjusted ``close``. Use ``adj_close``
(or an explicitly adjusted price column). Risk-free rates must be RBI 91-Day
T-Bill or MIBOR — never US Fed Funds / SOFR.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from src.data.corporate_actions import apply_adjustment_factor
from src.data.risk_free import attach_excess_returns, forward_fill_to_calendar

logger = logging.getLogger(__name__)

ReturnKind = Literal["simple", "log"]

RETURNS_COLUMNS: list[str] = [
    "trade_date",
    "symbol",
    "adj_close",
    "ret",
    "log_ret",
    "excess_ret",
    "daily_rf",
    "is_liquidated",
    "is_suspended",
]


def ensure_adj_close(
    df: pl.DataFrame,
    *,
    adj_factor_col: str = "adj_factor",
    close_col: str = "close",
) -> pl.DataFrame:
    """Guarantee an ``adj_close`` column exists.

    If missing, applies ``adj_factor`` (defaulting to 1.0) to ``close``.
    """
    if "adj_close" in df.columns:
        return df
    if adj_factor_col not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias(adj_factor_col))
        logger.warning("adj_factor missing — defaulting to 1.0 (document corporate-action coverage)")
    if close_col not in df.columns:
        raise KeyError(f"{close_col} required to build adj_close")
    return apply_adjustment_factor(df, price_cols=[close_col], factor_col=adj_factor_col)


def compute_price_returns(
    df: pl.DataFrame,
    *,
    price_col: str = "adj_close",
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """Vectorized simple and log returns from an adjusted price column.

    Liquidated / null prices yield null returns (never fabricated zeros).
    """
    if price_col not in df.columns:
        raise KeyError(
            f"{price_col} missing — run corporate-action adjustment before returns"
        )

    prev = pl.col(price_col).shift(1).over(symbol_col)
    simple = (pl.col(price_col) / prev - 1.0).alias("ret")
    log_ret = (pl.col(price_col) / prev).log().alias("log_ret")

    # Null-out returns on liquidated rows if flag present
    out = df.sort([symbol_col, date_col]).with_columns(simple, log_ret)
    if "is_liquidated" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("is_liquidated"))
            .then(pl.lit(None))
            .otherwise(pl.col("ret"))
            .alias("ret"),
            pl.when(pl.col("is_liquidated"))
            .then(pl.lit(None))
            .otherwise(pl.col("log_ret"))
            .alias("log_ret"),
        )
    return out


def attach_rf_excess(
    returns: pl.DataFrame,
    rf: pl.DataFrame,
    *,
    return_col: str = "ret",
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """Forward-fill India $R_f$ to the returns calendar and attach excess returns."""
    calendar = returns.get_column(date_col).unique().sort().to_list()
    rf_aligned = forward_fill_to_calendar(rf, calendar)
    # forward_fill uses `date`; attach_excess_returns expects rf.date
    return attach_excess_returns(
        returns, rf_aligned, return_col=return_col, date_col=date_col
    )


def build_returns_panel(
    clean_prices: pl.DataFrame,
    rf: pl.DataFrame | None = None,
    *,
    require_adj: bool = True,
) -> pl.DataFrame:
    """End-to-end: ensure adj_close → simple/log returns → optional excess returns."""
    prices = clean_prices
    if require_adj and "adj_close" not in prices.columns:
        prices = ensure_adj_close(prices)

    out = compute_price_returns(prices)
    if rf is not None and not rf.is_empty():
        out = attach_rf_excess(out, rf)
    elif "excess_ret" not in out.columns:
        out = out.with_columns(
            pl.lit(None).cast(pl.Float64).alias("excess_ret"),
            pl.lit(None).cast(pl.Float64).alias("daily_rf"),
        )

    keep = [c for c in RETURNS_COLUMNS if c in out.columns]
    # Preserve useful extras
    extras = [
        c
        for c in out.columns
        if c not in keep
        and c
        in {
            "series",
            "tottrdqty",
            "tottrdval",
            "adj_factor",
            "close",
            "is_missing_ohlc",
            "rf_source",
        }
    ]
    return out.select(keep + extras).sort(["trade_date", "symbol"])


def returns_to_numpy(
    df: pl.DataFrame,
    *,
    column: str = "ret",
) -> npt.NDArray[np.floating]:
    """Extract a float64 ndarray for econometric routines."""
    return df.get_column(column).to_numpy().astype(np.float64)
