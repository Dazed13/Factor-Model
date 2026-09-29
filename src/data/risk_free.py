"""India-local risk-free rate (RBI 91-Day T-Bill / MIBOR).

Never use US Fed Funds or SOFR. Annualized yields are converted to daily
simple returns via ``y / 365`` (actual/365 convention common for T-Bill
presentation in India).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl

from src.data.paths import (
    PROCESSED_RISK_FREE_DIR,
    RAW_RISK_FREE_DIR,
    ensure_data_dirs,
)

logger = logging.getLogger(__name__)

RfSource = Literal["rbi_91d_tbill", "mibor"]

RISK_FREE_COLUMNS: list[str] = [
    "date",
    "annualized_yield",
    "daily_rf",
    "source",
]


def annualized_yield_to_daily(
    annualized_yield: float | pl.Expr,
    *,
    day_count: int = 365,
) -> float | pl.Expr:
    """Convert an annualized decimal yield to a one-day simple return.

    Parameters
    ----------
    annualized_yield:
        e.g. ``0.065`` for 6.5%. Values ``> 1`` are treated as percent
        (``6.5`` -> ``0.065``) when passed as a scalar.
    day_count:
        India T-Bill presentation commonly uses actual/365.
    """
    if isinstance(annualized_yield, pl.Expr):
        y = (
            pl.when(annualized_yield > 1.0)
            .then(annualized_yield / 100.0)
            .otherwise(annualized_yield)
        )
        return y / pl.lit(float(day_count))

    y = float(annualized_yield)
    if y > 1.0:
        y /= 100.0
    return y / float(day_count)


def build_risk_free_frame(
    dates: list[date] | pl.Series,
    annualized_yields: list[float] | pl.Series,
    *,
    source: RfSource = "rbi_91d_tbill",
    day_count: int = 365,
) -> pl.DataFrame:
    """Construct a canonical risk-free panel from parallel date/yield arrays."""
    df = pl.DataFrame(
        {
            "date": dates,
            "annualized_yield": annualized_yields,
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    df = df.with_columns(
        (
            pl.when(pl.col("annualized_yield") > 1.0)
            .then(pl.col("annualized_yield") / 100.0)
            .otherwise(pl.col("annualized_yield"))
        ).alias("annualized_yield")
    )
    df = df.with_columns(
        (pl.col("annualized_yield") / pl.lit(float(day_count))).alias("daily_rf"),
        pl.lit(source).alias("source"),
    )
    return df.select(RISK_FREE_COLUMNS).sort("date")


def load_risk_free_csv(
    path: str | Path,
    *,
    source: RfSource = "rbi_91d_tbill",
    date_col: str = "date",
    yield_col: str = "annualized_yield",
    day_count: int = 365,
) -> pl.DataFrame:
    """Load a user-supplied RBI / MIBOR yield CSV.

    Accepts column aliases: ``date``/``Date``, and
    ``annualized_yield`` / ``yield`` / ``ytm`` / ``rate``.
    """
    # RBI exports sometimes mark missing auctions as "-".
    df = pl.read_csv(path, try_parse_dates=True, null_values=["-"])
    lower = {c.lower().strip(): c for c in df.columns}

    date_aliases = (date_col.lower(), "date", "trade_date", "auction_date")
    yield_aliases = (
        yield_col.lower(),
        "annualized_yield",
        "yield",
        "ytm",
        "rate",
        "tbill_91d",
        "mibor",
    )

    dcol = next((lower[a] for a in date_aliases if a in lower), None)
    ycol = next((lower[a] for a in yield_aliases if a in lower), None)
    if dcol is None or ycol is None:
        raise ValueError(
            f"Risk-free CSV must include date and yield columns; got {df.columns}"
        )

    return build_risk_free_frame(
        df.get_column(dcol).to_list(),
        df.get_column(ycol).cast(pl.Float64).to_list(),
        source=source,
        day_count=day_count,
    )


def forward_fill_to_calendar(
    rf: pl.DataFrame,
    calendar: list[date] | pl.Series,
) -> pl.DataFrame:
    """Align risk-free yields to a trading calendar via as-of forward fill."""
    cal = pl.DataFrame({"date": calendar}).with_columns(pl.col("date").cast(pl.Date))
    return (
        cal.join_asof(rf.sort("date"), on="date", strategy="backward")
        .select(RISK_FREE_COLUMNS)
        .sort("date")
    )


def attach_excess_returns(
    returns: pl.DataFrame,
    rf: pl.DataFrame,
    *,
    return_col: str = "ret",
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """Compute ``excess_ret = ret - daily_rf`` using India-local $R_f$."""
    if "daily_rf" not in rf.columns:
        raise KeyError("rf frame missing daily_rf")
    banned = {"fed_funds", "sofr", "us_tbill"}
    if "source" in rf.columns:
        sources = {str(s).lower() for s in rf.get_column("source").unique().to_list()}
        if sources & banned:
            raise ValueError(
                f"US risk-free sources are forbidden for NSE excess returns: {sources & banned}"
            )

    rf_join = rf.select(
        pl.col("date").alias(date_col),
        pl.col("daily_rf"),
        pl.col("source").alias("rf_source"),
    )
    out = returns.join(rf_join, on=date_col, how="left")
    return out.with_columns((pl.col(return_col) - pl.col("daily_rf")).alias("excess_ret"))


def save_risk_free(df: pl.DataFrame, name: str = "risk_free") -> Path:
    """Persist risk-free panel as snappy Parquet."""
    ensure_data_dirs()
    PROCESSED_RISK_FREE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RISK_FREE_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_RISK_FREE_DIR / f"{name}.parquet"
    df.write_parquet(out, compression="snappy")
    return out


def example_rbi_tbill_template(path: Path | None = None) -> Path:
    """Write a small example CSV template for manual RBI T-Bill yields."""
    ensure_data_dirs()
    path = path or (RAW_RISK_FREE_DIR / "rbi_91d_tbill_template.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "date": [date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 19)],
            "annualized_yield": [6.75, 6.78, 6.80],
            "source_note": ["RBI 91-Day T-Bill cut-off (percent)"] * 3,
        }
    ).write_csv(path)
    return path


def import_risk_free_file(
    path: str | Path,
    *,
    source: RfSource = "rbi_91d_tbill",
    dest_name: str = "risk_free",
) -> Path:
    """Normalize a user CSV/Parquet and write processed snappy Parquet + raw copy."""
    ensure_data_dirs()
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
        if "daily_rf" not in df.columns:
            raise ValueError("Parquet risk-free file must already include daily_rf")
    else:
        df = load_risk_free_csv(path, source=source)
        # Keep a raw copy for audit
        raw_out = RAW_RISK_FREE_DIR / f"{dest_name}.csv"
        df.select(["date", "annualized_yield"]).write_csv(raw_out)
    return save_risk_free(df, name=dest_name)


def import_risk_free_url(
    url: str,
    *,
    source: RfSource = "rbi_91d_tbill",
    dest_name: str = "risk_free",
    timeout: float = 60.0,
) -> Path:
    """Download a CSV from ``url`` and import via :func:`import_risk_free_file`."""
    import requests

    ensure_data_dirs()
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    raw_path = RAW_RISK_FREE_DIR / f"{dest_name}_download.csv"
    raw_path.write_bytes(resp.content)
    return import_risk_free_file(raw_path, source=source, dest_name=dest_name)
