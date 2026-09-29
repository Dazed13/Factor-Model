"""Idempotent Phase-2 pipeline: raw → clean → returns → fundamentals join.

Orchestrates Polars transforms and snappy Parquet / DuckDB persistence per
``.cursor/rules/python-perf.mdc``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl

from src.data.bhavcopy import load_bhavcopy_dir
from src.data.clean import apply_liquidation_policy, clean_bhavcopy, trading_calendar_from_panel
from src.data.corporate_actions import apply_adjustment_factor
from src.data.fundamentals import load_fundamentals_csv, point_in_time_fundamentals
from src.data.paths import (
    DUCKDB_PATH,
    PROCESSED_BHAVCOPY_DIR,
    PROCESSED_FUNDAMENTALS_DIR,
    PROCESSED_RETURNS_DIR,
    PROCESSED_RISK_FREE_DIR,
    RAW_BHAVCOPY_DIR,
    RAW_FUNDAMENTALS_DIR,
    RAW_RISK_FREE_DIR,
    ensure_data_dirs,
)
from src.data.returns import build_returns_panel
from src.data.risk_free import load_risk_free_csv, save_risk_free
from src.data.store import (
    connect,
    register_standard_views,
    write_partitioned_parquet,
)
from src.data.symbols import SymbolMap

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Paths and knobs for a reproducible rebuild."""

    raw_bhavcopy_dir: Path = RAW_BHAVCOPY_DIR
    processed_bhavcopy_dir: Path = PROCESSED_BHAVCOPY_DIR
    processed_returns_dir: Path = PROCESSED_RETURNS_DIR
    processed_fundamentals_dir: Path = PROCESSED_FUNDAMENTALS_DIR
    risk_free_csv: Path | None = None
    fundamentals_csv: Path | None = None
    duckdb_path: Path = DUCKDB_PATH
    apply_calendar: bool = True
    default_adj_factor: float = 1.0
    fundamental_lag_months: int = 3
    partition_by: str = "year"


@dataclass
class PipelineResult:
    """Artifacts produced by :func:`run_phase2_pipeline`."""

    clean_prices: pl.DataFrame
    returns: pl.DataFrame
    fundamentals: pl.DataFrame | None = None
    risk_free: pl.DataFrame | None = None
    bhavcopy_paths: list[Path] = field(default_factory=list)
    returns_paths: list[Path] = field(default_factory=list)
    duckdb_views: list[str] = field(default_factory=list)


def _resolve_risk_free(cfg: PipelineConfig) -> pl.DataFrame | None:
    candidates: list[Path] = []
    if cfg.risk_free_csv is not None:
        candidates.append(cfg.risk_free_csv)
    candidates.extend(sorted(RAW_RISK_FREE_DIR.glob("*.csv")))
    candidates.extend(sorted(PROCESSED_RISK_FREE_DIR.glob("*.parquet")))

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".parquet":
                return pl.read_parquet(path)
            return load_risk_free_csv(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping risk-free file %s: %s", path, exc)
    return None


def _resolve_fundamentals(cfg: PipelineConfig, symbol_map: SymbolMap) -> pl.DataFrame | None:
    candidates: list[Path] = []
    if cfg.fundamentals_csv is not None:
        candidates.append(cfg.fundamentals_csv)
    candidates.extend(sorted(RAW_FUNDAMENTALS_DIR.glob("*.csv")))
    candidates.extend(sorted(PROCESSED_FUNDAMENTALS_DIR.glob("*.parquet")))

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".parquet":
                return symbol_map.normalize_frame(pl.read_parquet(path), "symbol")
            return load_fundamentals_csv(
                path, lag_months=cfg.fundamental_lag_months, symbol_map=symbol_map
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping fundamentals file %s: %s", path, exc)
    return None


def join_fundamentals_asof(
    returns: pl.DataFrame,
    fundamentals: pl.DataFrame,
) -> pl.DataFrame:
    """Left-asof join fundamentals onto returns by symbol using available_date.

    For each (trade_date, symbol), attach the latest fundamental row with
    ``available_date <= trade_date`` (no look-ahead).
    """
    if returns.is_empty() or fundamentals.is_empty():
        return returns

    req = {"available_date", "symbol"}
    if not req.issubset(set(fundamentals.columns)):
        raise ValueError(f"fundamentals missing {req - set(fundamentals.columns)}")

    fund_cols = [
        c
        for c in fundamentals.columns
        if c in {"symbol", "available_date", "as_of_date", "market_cap", "book_value", "book_to_market"}
    ]
    fund = fundamentals.select(fund_cols).sort(["symbol", "available_date"])
    ret = returns.sort(["symbol", "trade_date"])
    return ret.join_asof(
        fund,
        left_on="trade_date",
        right_on="available_date",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    ).sort(["trade_date", "symbol"])


def run_phase2_pipeline(
    cfg: PipelineConfig | None = None,
    *,
    bhavcopy: pl.DataFrame | None = None,
    symbol_map: SymbolMap | None = None,
    register_duckdb: bool = True,
) -> PipelineResult:
    """Rebuild cleaned prices + returns panels idempotently.

    Parameters
    ----------
    bhavcopy:
        Optional in-memory panel (skips raw directory load) — useful for tests.
    """
    cfg = cfg or PipelineConfig()
    sm = symbol_map or SymbolMap()
    ensure_data_dirs()

    if bhavcopy is None:
        logger.info("Loading bhavcopy from %s", cfg.raw_bhavcopy_dir)
        raw = load_bhavcopy_dir(cfg.raw_bhavcopy_dir, symbol_map=sm)
    else:
        raw = bhavcopy

    if raw.is_empty():
        logger.warning("Empty bhavcopy input — nothing to process")
        return PipelineResult(clean_prices=raw, returns=raw)

    # Ensure adj_close exists (Phase-1 may have left raw closes only)
    if "adj_close" not in raw.columns:
        if "adj_factor" not in raw.columns:
            raw = raw.with_columns(pl.lit(cfg.default_adj_factor).alias("adj_factor"))
        raw = apply_adjustment_factor(raw, price_cols=["close"])

    calendar = trading_calendar_from_panel(raw) if cfg.apply_calendar else None
    clean = clean_bhavcopy(
        raw,
        symbol_map=sm,
        apply_calendar=cfg.apply_calendar,
        calendar=calendar,
    )
    if cfg.apply_calendar and calendar:
        # Re-apply liquidation on the cleaned unique panel using inferred calendar
        clean = apply_liquidation_policy(clean, calendar=calendar)

    rf = _resolve_risk_free(cfg)
    if rf is not None and not rf.is_empty():
        save_risk_free(rf)

    returns = build_returns_panel(clean, rf=rf)

    fundamentals = _resolve_fundamentals(cfg, sm)
    if fundamentals is not None and not fundamentals.is_empty():
        fund_to_store = fundamentals.with_columns(
            pl.col("available_date").alias("trade_date")
        )
        write_partitioned_parquet(
            fund_to_store,
            cfg.processed_fundamentals_dir,
            partition_by="year",  # type: ignore[arg-type]
            date_col="trade_date",
        )
        returns = join_fundamentals_asof(returns, fundamentals)

    bhav_paths = write_partitioned_parquet(
        clean,
        cfg.processed_bhavcopy_dir,
        partition_by=cfg.partition_by,  # type: ignore[arg-type]
    )
    ret_paths = write_partitioned_parquet(
        returns,
        cfg.processed_returns_dir,
        partition_by=cfg.partition_by,  # type: ignore[arg-type]
    )

    views: list[str] = []
    if register_duckdb:
        con = connect(cfg.duckdb_path)
        try:
            views = register_standard_views(con)
        finally:
            con.close()

    return PipelineResult(
        clean_prices=clean,
        returns=returns,
        fundamentals=fundamentals,
        risk_free=rf,
        bhavcopy_paths=bhav_paths,
        returns_paths=ret_paths,
        duckdb_views=views,
    )


def pit_fundamental_on_date(
    fundamentals: pl.DataFrame,
    as_of: date,
) -> pl.DataFrame:
    """Convenience wrapper for month-end PIT fundamental snapshots."""
    return point_in_time_fundamentals(fundamentals, as_of)
