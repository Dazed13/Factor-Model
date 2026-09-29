"""Corporate-action adjustment helpers.

Never use unadjusted closes for return computation. Adjusted closes are derived
from yfinance ``Adj Close`` / ``Close`` ratios (split + dividend / bonus effects
as provided by Yahoo) or from an explicit factor panel.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

import numpy as np
import numpy.typing as npt
import polars as pl

from src.data.symbols import SymbolMap, to_yfinance_symbols

logger = logging.getLogger(__name__)


def adjustment_factor_from_prices(
    close: npt.NDArray[np.floating],
    adj_close: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Element-wise ``adj_close / close`` with safe handling of zeros/NaNs."""
    close_f = np.asarray(close, dtype=np.float64)
    adj_f = np.asarray(adj_close, dtype=np.float64)
    factor = np.full_like(close_f, np.nan, dtype=np.float64)
    valid = np.isfinite(close_f) & np.isfinite(adj_f) & (close_f != 0.0)
    factor[valid] = adj_f[valid] / close_f[valid]
    return factor


def apply_adjustment_factor(
    prices: pl.DataFrame,
    *,
    price_cols: Iterable[str] = ("open", "high", "low", "close", "last", "prev_close"),
    factor_col: str = "adj_factor",
) -> pl.DataFrame:
    """Multiply OHLC columns by ``adj_factor`` and emit ``adj_*`` columns.

    Raw unadjusted columns are retained for audit; downstream returns must use
    ``adj_close`` (or other ``adj_*`` fields) only.
    """
    if factor_col not in prices.columns:
        raise KeyError(f"Missing adjustment factor column: {factor_col}")

    exprs = [
        (pl.col(c) * pl.col(factor_col)).alias(f"adj_{c}")
        for c in price_cols
        if c in prices.columns
    ]
    return prices.with_columns(exprs)


def attach_yfinance_adjustments(
    bhav: pl.DataFrame,
    *,
    symbols: list[str] | None = None,
    symbol_map: SymbolMap | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """Join yfinance Adj Close ratios onto a canonical bhavcopy panel.

    Parameters
    ----------
    bhav:
        Canonical panel with ``trade_date``, ``symbol``, ``close``.
    symbols:
        Optional subset of bare NSE symbols; defaults to unique symbols in ``bhav``.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for attach_yfinance_adjustments") from exc

    for name in ("yfinance", "peewee", "urllib3", "curl_cffi"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    sm = symbol_map or SymbolMap()
    syms = symbols or bhav.get_column("symbol").unique().to_list()
    yf_tickers = to_yfinance_symbols(syms, sm)

    if start is None:
        start = bhav.select(pl.col("trade_date").min()).item()
    if end is None:
        end = bhav.select(pl.col("trade_date").max()).item()
    if isinstance(start, date) and not hasattr(start, "hour"):
        start_s = start.isoformat()
    else:
        start_s = str(start)[:10]
    if isinstance(end, date) and not hasattr(end, "hour"):
        end_s = end.isoformat()
    else:
        end_s = str(end)[:10]

    logger.info("Downloading yfinance adjustments for %d tickers", len(yf_tickers))
    raw = yf.download(
        yf_tickers,
        start=start_s,
        end=end_s,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    frames: list[pl.DataFrame] = []
    # yfinance multi-ticker shape: columns MultiIndex (ticker, field)
    if getattr(raw.columns, "nlevels", 1) > 1:
        for yf_sym, bare in zip(yf_tickers, [sm.normalize_nse(s) for s in syms]):
            if yf_sym not in raw.columns.get_level_values(0):
                logger.warning("No yfinance data for %s", yf_sym)
                continue
            sub = raw[yf_sym].dropna(how="all").reset_index()
            date_col = "Date" if "Date" in sub.columns else sub.columns[0]
            part = pl.from_pandas(sub).rename({date_col: "trade_date"})
            part = part.with_columns(
                pl.col("trade_date").cast(pl.Datetime).dt.date().alias("trade_date"),
                pl.lit(bare).alias("symbol"),
            )
            if "Close" not in part.columns or "Adj Close" not in part.columns:
                continue
            part = part.with_columns(
                pl.Series(
                    "adj_factor",
                    adjustment_factor_from_prices(
                        part["Close"].to_numpy(),
                        part["Adj Close"].to_numpy(),
                    ),
                )
            ).select(["trade_date", "symbol", "adj_factor", "Adj Close"])
            frames.append(part.rename({"Adj Close": "yf_adj_close"}))
    else:
        # Single ticker
        bare = sm.normalize_nse(syms[0])
        sub = raw.dropna(how="all").reset_index()
        date_col = "Date" if "Date" in sub.columns else sub.columns[0]
        part = pl.from_pandas(sub).rename({date_col: "trade_date"})
        part = part.with_columns(
            pl.col("trade_date").cast(pl.Datetime).dt.date().alias("trade_date"),
            pl.lit(bare).alias("symbol"),
            pl.Series(
                "adj_factor",
                adjustment_factor_from_prices(
                    part["Close"].to_numpy(),
                    part["Adj Close"].to_numpy(),
                ),
            ),
        ).select(["trade_date", "symbol", "adj_factor", "Adj Close"])
        frames.append(part.rename({"Adj Close": "yf_adj_close"}))

    if not frames:
        logger.warning("No yfinance adjustment rows; setting adj_factor=1.0")
        return apply_adjustment_factor(
            bhav.with_columns(pl.lit(1.0).alias("adj_factor"))
        )

    adj = pl.concat(frames, how="vertical_relaxed")
    merged = bhav.join(adj, on=["trade_date", "symbol"], how="left")
    merged = merged.with_columns(pl.col("adj_factor").fill_null(1.0))
    return apply_adjustment_factor(merged)


def adjusted_returns(
    df: pl.DataFrame,
    *,
    price_col: str = "adj_close",
    symbol_col: str = "symbol",
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """Compute simple returns from an adjusted price column (never raw close)."""
    if price_col not in df.columns:
        raise KeyError(
            f"{price_col} missing — run corporate-action adjustment before returns"
        )
    return (
        df.sort([symbol_col, date_col])
        .with_columns(
            (pl.col(price_col) / pl.col(price_col).shift(1).over(symbol_col) - 1.0).alias(
                "ret"
            )
        )
    )
