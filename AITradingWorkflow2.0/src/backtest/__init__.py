"""
Backtest Module

Backtesting framework with:
- Walk-forward validation
- Realistic transaction cost modeling
- Performance metrics and attribution
"""

from .engine import (
    Trade,
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    WalkForwardPeriod,
    BacktestEngine,
    FillAssumption,
)
from .costs import (
    CostConfig,
    TransactionCostCalculator,
    calculate_cost,
    DEFAULT_COSTS,
)
from .metrics import (
    PerformanceMetrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_cagr,
    calculate_var,
    bootstrap_sharpe_ci,
    calculate_trade_statistics,
    calculate_all_metrics,
)

__all__ = [
    # Engine
    "Trade",
    "BacktestConfig",
    "BacktestMetrics",
    "BacktestResult",
    "WalkForwardPeriod",
    "BacktestEngine",
    "FillAssumption",
    
    # Costs
    "CostConfig",
    "TransactionCostCalculator",
    "calculate_cost",
    "DEFAULT_COSTS",
    
    # Metrics
    "PerformanceMetrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_cagr",
    "calculate_var",
    "bootstrap_sharpe_ci",
    "calculate_trade_statistics",
    "calculate_all_metrics",
]
