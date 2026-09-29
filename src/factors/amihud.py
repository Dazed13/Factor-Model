"""Amihud (2002) illiquidity characteristic.

Daily ratio:

    ILLIQ_{i,t} = |R_{i,t}| / Volume_{i,t}^{INR}

Zero / missing volume → NaN (never ZeroDivisionError).

Tradable signal is the D-day rolling mean lagged by one day:

    Signal_{i,t}^{trade} = mean(ILLIQ_{i,t-D+1:t})_{shifted by 1}
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

DEFAULT_WINDOW: int = 30


def daily_amihud_illiq(
    ret: npt.NDArray[np.floating] | float,
    volume_inr: npt.NDArray[np.floating] | float,
    *,
    zero_volume: float = np.nan,
) -> npt.NDArray[np.floating] | float:
    """Element-wise Amihud ratio with safe zero-volume handling.

    Parameters
    ----------
    zero_volume:
        Value when volume <= 0 or non-finite. Default ``np.nan`` (preferred).
        ``0.0`` is also allowed by project rules.
    """
    r = np.asarray(ret, dtype=np.float64)
    v = np.asarray(volume_inr, dtype=np.float64)
    scalar_out = r.ndim == 0 and v.ndim == 0
    r_b, v_b = np.broadcast_arrays(np.atleast_1d(r), np.atleast_1d(v))
    out = np.full(r_b.shape, zero_volume, dtype=np.float64)
    valid = np.isfinite(r_b) & np.isfinite(v_b) & (v_b > 0)
    out[valid] = np.abs(r_b[valid]) / v_b[valid]
    if scalar_out:
        return float(out.reshape(()))
    return out


def attach_daily_illiq(
    df: pl.DataFrame,
    *,
    ret_col: str = "ret",
    volume_col: str = "tottrdval",
    out_col: str = "illiq",
) -> pl.DataFrame:
    """Vectorized daily Amihud ILLIQ; zero volume → null."""
    if ret_col not in df.columns:
        raise KeyError(f"Missing return column: {ret_col}")
    if volume_col not in df.columns:
        raise KeyError(f"Missing volume column: {volume_col}")

    return df.with_columns(
        pl.when(
            pl.col(volume_col).is_null()
            | pl.col(volume_col).is_nan()
            | (pl.col(volume_col) <= 0)
            | pl.col(ret_col).is_null()
            | pl.col(ret_col).is_nan()
        )
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(pl.col(ret_col).abs() / pl.col(volume_col))
        .alias(out_col)
    )


def attach_rolling_illiq(
    df: pl.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    illiq_col: str = "illiq",
    out_col: str = "illiq_roll",
    min_periods: int | None = None,
) -> pl.DataFrame:
    """D-day rolling mean of daily ILLIQ."""
    if window <= 0:
        raise ValueError("window must be positive")
    min_p = min_periods if min_periods is not None else max(1, window // 2)

    if illiq_col not in df.columns:
        df = attach_daily_illiq(df)

    return df.sort([symbol_col, date_col]).with_columns(
        pl.col(illiq_col)
        .rolling_mean(window_size=window, min_periods=min_p)
        .over(symbol_col)
        .alias(out_col)
    )


def attach_illiq_trade_signal(
    df: pl.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
    signal_col: str = "illiq_signal",
) -> pl.DataFrame:
    """Tradable ILLIQ signal = rolling mean shifted by 1 (no lookahead)."""
    out = attach_rolling_illiq(
        df if "illiq" in df.columns else attach_daily_illiq(df),
        window=window,
        symbol_col=symbol_col,
        date_col=date_col,
    )
    return out.with_columns(
        pl.col("illiq_roll").shift(1).over(symbol_col).alias(signal_col)
    )


def build_amihud_characteristic(
    returns: pl.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
) -> pl.DataFrame:
    """Convenience: returns panel → daily / rolling / lagged ILLIQ columns."""
    return attach_illiq_trade_signal(returns, window=window)
