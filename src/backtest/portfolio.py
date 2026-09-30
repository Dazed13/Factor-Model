"""Dollar-neutral long–short portfolio construction.

Weight convention
-----------------
On each formation date, long-sleeve weights sum to ``+0.5`` and short-sleeve
weights sum to ``-0.5``, so:

* ``sum(w) == 0`` (dollar neutral)
* gross exposure ``sum(|w|) == 1``
* long weights strictly ``> 0``, short weights strictly ``< 0``

Supports equal-weight or value-weight (by ``market_cap`` / ``me``) within sleeves.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from src.backtest.sorts import N_BUCKETS, SortKind, assign_sorts

Weighting = Literal["equal", "value"]

WEIGHT_SUM_TOL: float = 1e-10


def _raw_weights(df: pl.DataFrame, weighting: Weighting, weight_col: str | None) -> pl.Expr:
    if weighting == "value" and weight_col and weight_col in df.columns:
        # Keep nulls as null — never coerce missing ME to 0 (that creates zero weights).
        return pl.col(weight_col).cast(pl.Float64)
    return pl.lit(1.0)


def build_long_short_weights(
    sorted_panel: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    quantile_col: str = "quantile",
    n_buckets: int = 5,
    long_quantile: int | None = None,
    short_quantile: int | None = None,
    long_high: bool = True,
    weighting: Weighting = "equal",
    weight_col: str | None = "market_cap",
) -> pl.DataFrame:
    """Form dollar-neutral weights from quantile assignments.

    Parameters
    ----------
    long_high:
        If True, long the top quantile (N) and short quantile 1.
        If False, long quantile 1 and short the top (e.g. size: long small).
    """
    top = n_buckets
    bottom = 1
    if long_quantile is None:
        long_quantile = top if long_high else bottom
    if short_quantile is None:
        short_quantile = bottom if long_high else top

    work = sorted_panel.filter(pl.col(quantile_col).is_not_null()).with_columns(
        _raw_weights(sorted_panel, weighting, weight_col).alias("_raw_w"),
        (pl.col(quantile_col) == long_quantile).alias("_is_long"),
        (pl.col(quantile_col) == short_quantile).alias("_is_short"),
    )
    work = work.filter(pl.col("_is_long") | pl.col("_is_short"))

    # Value-weighting requires strictly positive ME; drop missing/non-positive.
    if weighting == "value":
        work = work.filter(
            pl.col("_raw_w").is_not_null() & (pl.col("_raw_w") > 0)
        )
    else:
        work = work.with_columns(pl.col("_raw_w").fill_null(1.0))

    # Sleeve totals per date
    long_tot = (
        work.filter(pl.col("_is_long"))
        .group_by(date_col)
        .agg(pl.col("_raw_w").sum().alias("_long_tot"))
    )
    short_tot = (
        work.filter(pl.col("_is_short"))
        .group_by(date_col)
        .agg(pl.col("_raw_w").sum().alias("_short_tot"))
    )
    work = work.join(long_tot, on=date_col, how="left").join(
        short_tot, on=date_col, how="left"
    )

    weights = work.with_columns(
        pl.when(pl.col("_is_long") & (pl.col("_long_tot") > 0))
        .then(0.5 * pl.col("_raw_w") / pl.col("_long_tot"))
        .when(pl.col("_is_short") & (pl.col("_short_tot") > 0))
        .then(-0.5 * pl.col("_raw_w") / pl.col("_short_tot"))
        .otherwise(None)
        .alias("weight")
    ).filter(pl.col("weight").is_not_null() & (pl.col("weight") != 0))

    return weights.select(
        [
            date_col,
            symbol_col,
            quantile_col,
            "weight",
            pl.col("_is_long").alias("is_long"),
            pl.col("_is_short").alias("is_short"),
        ]
        + ([weight_col] if weight_col and weight_col in weights.columns else [])
    )


def assert_dollar_neutral(
    weights: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    tol: float = WEIGHT_SUM_TOL,
) -> None:
    """Raise ``AssertionError`` if any date violates dollar-neutrality / sign rules."""
    if weights.is_empty():
        return

    # Global sign constraints
    longs = weights.filter(pl.col("weight") > 0)
    shorts = weights.filter(pl.col("weight") < 0)
    zeros = weights.filter(pl.col("weight") == 0)
    if zeros.height > 0:
        raise AssertionError("Zero weights are not allowed in active sleeves")
    if longs.height and (longs["weight"] <= 0).any():
        raise AssertionError("Long-sleeve weights must be strictly positive")
    if shorts.height and (shorts["weight"] >= 0).any():
        raise AssertionError("Short-sleeve weights must be strictly negative")

    sums = weights.group_by(date_col).agg(pl.col("weight").sum().alias("w_sum"))
    bad = sums.filter(pl.col("w_sum").abs() >= tol)
    if bad.height > 0:
        raise AssertionError(
            f"Dollar neutrality violated on {bad.height} dates; "
            f"max |sum|={sums['w_sum'].abs().max()}"
        )


def weights_sum_by_date(
    weights: pl.DataFrame,
    *,
    date_col: str = "trade_date",
) -> pl.DataFrame:
    return weights.group_by(date_col).agg(
        pl.col("weight").sum().alias("w_sum"),
        pl.col("weight").abs().sum().alias("gross"),
        (pl.col("weight") > 0).sum().alias("n_long"),
        (pl.col("weight") < 0).sum().alias("n_short"),
    ).sort(date_col)


def form_quantile_portfolio(
    panel: pl.DataFrame,
    signal_col: str,
    *,
    kind: SortKind = "quintile",
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    long_high: bool = True,
    weighting: Weighting = "equal",
    weight_col: str | None = "market_cap",
    validate: bool = True,
) -> pl.DataFrame:
    """Sort → dollar-neutral long/short weights in one call."""
    n = N_BUCKETS[kind]
    sorted_panel = assign_sorts(
        panel, signal_col, date_col=date_col, kind=kind, out_col="quantile"
    )
    weights = build_long_short_weights(
        sorted_panel,
        date_col=date_col,
        symbol_col=symbol_col,
        n_buckets=n,
        long_high=long_high,
        weighting=weighting,
        weight_col=weight_col,
    )
    if validate:
        assert_dollar_neutral(weights, date_col=date_col)
    return weights


def weights_to_numpy(weights: pl.DataFrame) -> npt.NDArray[np.floating]:
    return weights.get_column("weight").to_numpy().astype(np.float64)
