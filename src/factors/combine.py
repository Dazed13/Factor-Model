"""Assemble characteristic panels + factor return series; persist to Parquet."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from src.data.paths import PROCESSED_FACTORS_DIR, ensure_data_dirs
from src.data.store import write_partitioned_parquet
from src.factors.amihud import DEFAULT_WINDOW, build_amihud_characteristic
from src.factors.momentum import (
    attach_momentum_characteristic,
    construct_illiq_factor,
    construct_wml,
)
from src.factors.size import attach_size_characteristic, construct_smb
from src.factors.value import attach_value_characteristic, construct_hml
from src.factors.winsorize import prepare_signal

logger = logging.getLogger(__name__)


@dataclass
class FactorBuildResult:
    """Outputs of :func:`build_factors`."""

    characteristics: pl.DataFrame
    factor_returns: pl.DataFrame
    characteristic_paths: list[Path] = field(default_factory=list)
    factor_paths: list[Path] = field(default_factory=list)


def build_characteristic_panel(
    returns: pl.DataFrame,
    *,
    amihud_window: int = DEFAULT_WINDOW,
    winsorize: bool = True,
) -> pl.DataFrame:
    """Attach Size, Value, Momentum, and Amihud characteristics (+ optional winsor/z)."""
    panel = returns
    panel = build_amihud_characteristic(panel, window=amihud_window)

    if "market_cap" in panel.columns or "me" in panel.columns:
        panel = attach_size_characteristic(panel)
        sig_col = "log_me" if "log_me" in panel.columns else "me"
        if winsorize:
            panel = prepare_signal(panel, sig_col, winsor=True, zscore=True)

    if "book_to_market" in panel.columns or "btm" in panel.columns:
        panel = attach_value_characteristic(panel)
        if winsorize:
            panel = prepare_signal(panel, "btm", winsor=True, zscore=True)

    panel = attach_momentum_characteristic(panel)
    if winsorize:
        panel = prepare_signal(panel, "mom_12_1", winsor=True, zscore=True)
        if "illiq_signal" in panel.columns:
            panel = prepare_signal(panel, "illiq_signal", winsor=True, zscore=True)

    return panel.sort(["trade_date", "symbol"])


def build_factor_returns(characteristics: pl.DataFrame) -> pl.DataFrame:
    """Construct SMB, HML, WML, ILLIQ daily factor return series."""
    frames: list[pl.DataFrame] = []

    if "me" in characteristics.columns or "market_cap" in characteristics.columns:
        smb = construct_smb(characteristics)
        frames.append(smb)

    if "btm" in characteristics.columns or "book_to_market" in characteristics.columns:
        hml = construct_hml(characteristics)
        frames.append(hml)

    if "mom_12_1" in characteristics.columns:
        wml = construct_wml(characteristics)
        frames.append(wml)

    if "illiq_signal" in characteristics.columns:
        illiq = construct_illiq_factor(characteristics)
        frames.append(illiq)

    if not frames:
        return pl.DataFrame({"trade_date": pl.Series([], dtype=pl.Date)})

    out = frames[0]
    for fr in frames[1:]:
        out = out.join(fr, on="trade_date", how="outer", coalesce=True)
    return out.sort("trade_date")


def write_factor_artifacts(
    characteristics: pl.DataFrame,
    factor_returns: pl.DataFrame,
    *,
    dest_dir: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    """Persist characteristics (by year) and factor returns under processed/factors."""
    ensure_data_dirs()
    root = dest_dir or PROCESSED_FACTORS_DIR
    char_dir = root / "characteristics"
    fac_dir = root / "factor_returns"

    char_paths = write_partitioned_parquet(
        characteristics, char_dir, partition_by="year", date_col="trade_date"
    )
    # Factor returns are a narrow daily panel — single parquet is fine; still year-partition
    fac_paths = write_partitioned_parquet(
        factor_returns, fac_dir, partition_by="year", date_col="trade_date"
    )
    return char_paths, fac_paths


def build_factors(
    returns: pl.DataFrame,
    *,
    amihud_window: int = DEFAULT_WINDOW,
    winsorize: bool = True,
    dest_dir: Path | None = None,
    persist: bool = True,
) -> FactorBuildResult:
    """End-to-end Phase-3 factor construction from a returns panel."""
    if returns.is_empty():
        logger.warning("Empty returns panel — skipping factor build")
        return FactorBuildResult(characteristics=returns, factor_returns=returns)

    chars = build_characteristic_panel(
        returns, amihud_window=amihud_window, winsorize=winsorize
    )
    factors = build_factor_returns(chars)

    char_paths: list[Path] = []
    fac_paths: list[Path] = []
    if persist:
        char_paths, fac_paths = write_factor_artifacts(
            chars, factors, dest_dir=dest_dir
        )
        logger.info(
            "Wrote %d characteristic partitions, %d factor partitions",
            len(char_paths),
            len(fac_paths),
        )

    return FactorBuildResult(
        characteristics=chars,
        factor_returns=factors,
        characteristic_paths=char_paths,
        factor_paths=fac_paths,
    )
