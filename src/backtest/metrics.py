"""
Backtest Metrics Module

Performance metrics and statistical calculations for backtests.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    
    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sharpe_ci_low: Optional[float] = None
    sharpe_ci_high: Optional[float] = None
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    var_95: float = 0.0  # Value at Risk
    
    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    expectancy: float = 0.0


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio (downside risk only).
    
    Args:
        returns: Series of returns
        target_return: Minimum acceptable return
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Downside returns
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        return np.inf  # No downside
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std == 0:
        return 0.0
    
    sortino = (returns.mean() - target_return) / downside_std * np.sqrt(periods_per_year)
    
    return sortino


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int]:
    """
    Calculate maximum drawdown and duration.
    
    Args:
        equity_curve: Series of portfolio values
    
    Returns:
        Tuple of (max_drawdown_pct, duration_days)
    """
    if len(equity_curve) == 0:
        return 0.0, 0
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown series
    drawdown = (equity_curve - running_max) / running_max
    
    # Maximum drawdown
    max_dd = abs(drawdown.min())
    
    # Drawdown duration
    is_in_drawdown = drawdown < 0
    
    max_duration = 0
    current_duration = 0
    
    for in_dd in is_in_drawdown:
        if in_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_dd, max_duration


def calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float
) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
        years: Number of years
    
    Returns:
        CAGR as decimal
    """
    if initial_value <= 0 or years <= 0:
        return 0.0
    
    return (final_value / initial_value) ** (1 / years) - 1


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Calculate Value at Risk.
    
    Args:
        returns: Series of returns
        confidence: Confidence level (e.g., 0.95)
    
    Returns:
        VaR as positive number (potential loss)
    """
    if len(returns) == 0:
        return 0.0
    
    var = abs(np.percentile(returns, (1 - confidence) * 100))
    return var


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_samples: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for Sharpe ratio.
    
    Args:
        returns: Series of returns
        n_samples: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(returns) < 10:
        return (0.0, 0.0)
    
    sharpe_samples = []
    
    for _ in range(n_samples):
        # Sample with replacement
        sample = returns.sample(n=len(returns), replace=True)
        sharpe = calculate_sharpe_ratio(sample)
        sharpe_samples.append(sharpe)
    
    sharpe_samples = np.array(sharpe_samples)
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(sharpe_samples, alpha * 100)
    upper = np.percentile(sharpe_samples, (1 - alpha) * 100)
    
    return (lower, upper)


def calculate_trade_statistics(
    pnls: List[float]
) -> dict:
    """
    Calculate trade-level statistics.
    
    Args:
        pnls: List of trade P&Ls
    
    Returns:
        Dictionary of statistics
    """
    if not pnls:
        return {
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
        }
    
    pnls = np.array(pnls)
    
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    
    total_trades = len(pnls)
    win_rate = len(winners) / total_trades if total_trades > 0 else 0
    
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0
    
    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    return {
        "total_trades": total_trades,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def calculate_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    trade_pnls: List[float],
    years: float,
    initial_capital: float
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.
    
    Args:
        returns: Series of period returns
        equity_curve: Series of portfolio values
        trade_pnls: List of trade P&Ls
        years: Number of years in backtest
        initial_capital: Starting capital
    
    Returns:
        PerformanceMetrics dataclass
    """
    metrics = PerformanceMetrics()
    
    # Returns
    final_value = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    metrics.total_return = (final_value - initial_capital) / initial_capital
    metrics.cagr = calculate_cagr(initial_capital, final_value, years) if years > 0 else 0
    
    # Risk-adjusted
    metrics.sharpe_ratio = calculate_sharpe_ratio(returns)
    metrics.sortino_ratio = calculate_sortino_ratio(returns)
    
    # Bootstrap CI for Sharpe
    if len(returns) >= 30:
        ci_low, ci_high = bootstrap_sharpe_ci(returns)
        metrics.sharpe_ci_low = ci_low
        metrics.sharpe_ci_high = ci_high
    
    # Risk
    metrics.volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
    metrics.max_drawdown, metrics.max_drawdown_duration_days = calculate_max_drawdown(equity_curve)
    metrics.var_95 = calculate_var(returns)
    
    # Calmar ratio
    if metrics.max_drawdown > 0:
        metrics.calmar_ratio = metrics.cagr / metrics.max_drawdown
    
    # Trade statistics
    trade_stats = calculate_trade_statistics(trade_pnls)
    metrics.total_trades = trade_stats["total_trades"]
    metrics.win_rate = trade_stats["win_rate"]
    metrics.profit_factor = trade_stats["profit_factor"]
    metrics.avg_win = trade_stats["avg_win"]
    metrics.avg_loss = trade_stats["avg_loss"]
    metrics.expectancy = trade_stats["expectancy"]
    
    if metrics.avg_loss > 0:
        metrics.win_loss_ratio = metrics.avg_win / metrics.avg_loss
    
    return metrics
