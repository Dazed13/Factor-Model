"""Market-data ingestion + Phase-2 engineering package."""

from src.data.bhavcopy import (
    ALLOWED_SERIES,
    CANONICAL_COLUMNS,
    download_bhavcopy,
    download_bhavcopy_range,
    load_bhavcopy_dir,
    load_bhavcopy_file,
    parse_bhavcopy,
    parse_old_bhavcopy,
    parse_udiff_bhavcopy,
    write_bhavcopy_parquet,
)
from src.data.clean import apply_liquidation_policy, clean_bhavcopy
from src.data.corporate_actions import (
    adjusted_returns,
    adjustment_factor_from_prices,
    apply_adjustment_factor,
    attach_yfinance_adjustments,
)
from src.data.fundamentals import (
    apply_reporting_lag,
    fetch_yfinance_fundamentals,
    lag_available_date,
    load_fundamentals_csv,
    point_in_time_fundamentals,
    save_fundamentals,
)
from src.data.market_equity import (
    build_market_equity_panel,
    fetch_shares_history,
    market_equity_asof,
    merge_me_into_fundamentals,
)
from src.data.screener import (
    fetch_screener_book_equity,
    merge_book_into_fundamentals,
    parse_balance_sheet_table,
)
from src.data.paths import ensure_data_dirs
from src.data.pipeline import PipelineConfig, PipelineResult, run_phase2_pipeline
from src.data.returns import build_returns_panel, compute_price_returns
from src.data.risk_free import (
    annualized_yield_to_daily,
    attach_excess_returns,
    build_risk_free_frame,
    load_risk_free_csv,
    save_risk_free,
)
from src.data.store import (
    connect,
    read_partitioned_parquet,
    register_standard_views,
    write_partitioned_parquet,
)
from src.data.symbols import SymbolMap, strip_exchange_suffix, to_yfinance_symbols
from src.data.universe import (
    fetch_nifty500_constituents,
    load_universe_snapshots,
    point_in_time_membership,
    save_universe_snapshot,
)

__all__ = [
    "ALLOWED_SERIES",
    "CANONICAL_COLUMNS",
    "PipelineConfig",
    "PipelineResult",
    "SymbolMap",
    "adjusted_returns",
    "adjustment_factor_from_prices",
    "annualized_yield_to_daily",
    "apply_adjustment_factor",
    "apply_liquidation_policy",
    "apply_reporting_lag",
    "attach_excess_returns",
    "attach_yfinance_adjustments",
    "build_market_equity_panel",
    "build_returns_panel",
    "build_risk_free_frame",
    "clean_bhavcopy",
    "compute_price_returns",
    "connect",
    "download_bhavcopy",
    "download_bhavcopy_range",
    "ensure_data_dirs",
    "fetch_nifty500_constituents",
    "fetch_screener_book_equity",
    "fetch_shares_history",
    "fetch_yfinance_fundamentals",
    "lag_available_date",
    "market_equity_asof",
    "merge_book_into_fundamentals",
    "merge_me_into_fundamentals",
    "parse_balance_sheet_table",
    "load_bhavcopy_dir",
    "load_bhavcopy_file",
    "load_fundamentals_csv",
    "load_risk_free_csv",
    "load_universe_snapshots",
    "parse_bhavcopy",
    "parse_old_bhavcopy",
    "parse_udiff_bhavcopy",
    "point_in_time_fundamentals",
    "point_in_time_membership",
    "read_partitioned_parquet",
    "register_standard_views",
    "run_phase2_pipeline",
    "save_fundamentals",
    "save_risk_free",
    "save_universe_snapshot",
    "strip_exchange_suffix",
    "to_yfinance_symbols",
    "write_bhavcopy_parquet",
    "write_partitioned_parquet",
]
