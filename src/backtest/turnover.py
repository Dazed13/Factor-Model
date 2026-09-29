"""Portfolio turnover between rebalance dates."""

from __future__ import annotations

import polars as pl


def compute_turnover(
    weights: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    weight_col: str = "weight",
) -> pl.DataFrame:
    """One-way and two-way turnover across consecutive rebalance dates.

    Definitions
    -----------
    * **two-way** (standard): ``0.5 * sum_i |w_{i,t} - w_{i,t-1}|``
    * **one-way**: ``sum_i max(w_{i,t} - w_{i,t-1}, 0)`` (purchases only)

    Missing names at either date are treated as weight 0.
    """
    if weights.is_empty():
        return pl.DataFrame(
            schema={
                date_col: pl.Date,
                "turnover_one_way": pl.Float64,
                "turnover_two_way": pl.Float64,
            }
        )

    dates = weights.get_column(date_col).unique().sort().to_list()
    if len(dates) < 2:
        return pl.DataFrame(
            {
                date_col: dates,
                "turnover_one_way": [0.0] * len(dates),
                "turnover_two_way": [0.0] * len(dates),
            }
        )

    w = weights.select([date_col, symbol_col, weight_col]).unique(
        subset=[date_col, symbol_col], keep="last"
    )

    rows: list[dict] = []
    prev = w.filter(pl.col(date_col) == dates[0]).select(
        [symbol_col, pl.col(weight_col).alias("w_prev")]
    )
    rows.append(
        {date_col: dates[0], "turnover_one_way": 0.0, "turnover_two_way": 0.0}
    )

    for d in dates[1:]:
        cur = w.filter(pl.col(date_col) == d).select(
            [symbol_col, pl.col(weight_col).alias("w_cur")]
        )
        merged = prev.join(cur, on=symbol_col, how="full", coalesce=True).with_columns(
            pl.col("w_prev").fill_null(0.0),
            pl.col("w_cur").fill_null(0.0),
        )
        delta = merged.with_columns((pl.col("w_cur") - pl.col("w_prev")).alias("dw"))
        two_way = 0.5 * float(delta.select(pl.col("dw").abs().sum()).item())
        one_way = float(
            delta.select(pl.when(pl.col("dw") > 0).then(pl.col("dw")).otherwise(0.0).sum()).item()
        )
        rows.append(
            {
                date_col: d,
                "turnover_one_way": one_way,
                "turnover_two_way": two_way,
            }
        )
        prev = cur.rename({"w_cur": "w_prev"})

    return pl.DataFrame(rows).sort(date_col)


def average_turnover(turnover: pl.DataFrame) -> dict[str, float]:
    """Mean one-way / two-way turnover excluding the first (zero) rebalance."""
    if turnover.height <= 1:
        return {"turnover_one_way": 0.0, "turnover_two_way": 0.0}
    tail = turnover.slice(1)
    return {
        "turnover_one_way": float(tail["turnover_one_way"].mean()),
        "turnover_two_way": float(tail["turnover_two_way"].mean()),
    }
