"""Shared path constants for market-data artifacts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

RAW_BHAVCOPY_DIR: Path = RAW_DIR / "bhavcopy"
RAW_UNIVERSE_DIR: Path = RAW_DIR / "universe"
RAW_FUNDAMENTALS_DIR: Path = RAW_DIR / "fundamentals"
RAW_RISK_FREE_DIR: Path = RAW_DIR / "risk_free"

PROCESSED_BHAVCOPY_DIR: Path = PROCESSED_DIR / "bhavcopy"
PROCESSED_RETURNS_DIR: Path = PROCESSED_DIR / "returns"
PROCESSED_UNIVERSE_DIR: Path = PROCESSED_DIR / "universe"
PROCESSED_FUNDAMENTALS_DIR: Path = PROCESSED_DIR / "fundamentals"
PROCESSED_RISK_FREE_DIR: Path = PROCESSED_DIR / "risk_free"
PROCESSED_FACTORS_DIR: Path = PROCESSED_DIR / "factors"
PROCESSED_REPORTS_DIR: Path = PROCESSED_DIR / "reports"

DUCKDB_PATH: Path = PROCESSED_DIR / "factor_model.duckdb"


def ensure_data_dirs() -> None:
    """Create raw/processed subdirectories if missing."""
    for path in (
        RAW_BHAVCOPY_DIR,
        RAW_UNIVERSE_DIR,
        RAW_FUNDAMENTALS_DIR,
        RAW_RISK_FREE_DIR,
        PROCESSED_BHAVCOPY_DIR,
        PROCESSED_RETURNS_DIR,
        PROCESSED_UNIVERSE_DIR,
        PROCESSED_FUNDAMENTALS_DIR,
        PROCESSED_RISK_FREE_DIR,
        PROCESSED_FACTORS_DIR,
        PROCESSED_REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
