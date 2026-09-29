"""Portfolio sorts, backtesting, Fama–MacBeth, and performance metrics."""

from src.backtest.engine import BacktestConfig, BacktestResult, run_backtest, run_quantile_spread
from src.backtest.fama_macbeth import (
    FMResult,
    fama_macbeth_characteristics,
    fama_macbeth_two_pass,
    newey_west_mean_se,
)
from src.backtest.metrics import (
    annualized_sharpe,
    max_drawdown,
    quantile_metrics,
    summarize_returns,
)
from src.backtest.portfolio import (
    assert_dollar_neutral,
    form_quantile_portfolio,
    weights_sum_by_date,
)
from src.backtest.report import export_backtest_report, export_fm_report
from src.backtest.sorts import assign_sorts, filter_eligible, lag_signal, rebalance_dates
from src.backtest.tearsheet import TearsheetResult, build_tearsheet
from src.backtest.turnover import average_turnover, compute_turnover

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "FMResult",
    "TearsheetResult",
    "annualized_sharpe",
    "assert_dollar_neutral",
    "assign_sorts",
    "average_turnover",
    "build_tearsheet",
    "compute_turnover",
    "export_backtest_report",
    "export_fm_report",
    "fama_macbeth_characteristics",
    "fama_macbeth_two_pass",
    "filter_eligible",
    "form_quantile_portfolio",
    "lag_signal",
    "max_drawdown",
    "newey_west_mean_se",
    "quantile_metrics",
    "rebalance_dates",
    "run_backtest",
    "run_quantile_spread",
    "summarize_returns",
    "weights_sum_by_date",
]
