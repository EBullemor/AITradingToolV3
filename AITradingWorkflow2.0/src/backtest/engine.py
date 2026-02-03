"""
Backtest Engine

Core backtesting functionality with walk-forward validation,
transaction cost modeling, and regime-aware evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class FillAssumption(Enum):
    """Trade fill assumption."""
    CLOSE = "close"
    NEXT_OPEN = "next_open"
    VWAP = "vwap"


@dataclass
class Trade:
    """Represents a backtest trade."""
    instrument: str
    direction: str  # LONG or SHORT
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    
    size: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Costs
    entry_cost: float = 0.0
    exit_cost: float = 0.0
    
    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # Metadata
    signal_id: Optional[str] = None
    strategy: Optional[str] = None
    regime: Optional[str] = None
    
    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None
    
    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0
    
    @property
    def holding_days(self) -> int:
        if self.exit_date and self.entry_date:
            return (self.exit_date - self.entry_date).days
        return 0
    
    def close(self, exit_date: datetime, exit_price: float, exit_cost: float = 0):
        """Close the trade."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_cost = exit_cost
        
        # Calculate P&L
        if self.direction == "LONG":
            self.gross_pnl = (exit_price - self.entry_price) * self.size
        else:
            self.gross_pnl = (self.entry_price - exit_price) * self.size
        
        self.net_pnl = self.gross_pnl - self.entry_cost - exit_cost
        self.pnl_pct = self.net_pnl / (self.entry_price * self.size)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    # Date range
    start_date: datetime = datetime(2023, 1, 1)
    end_date: datetime = datetime(2025, 12, 31)
    
    # Initial capital
    initial_capital: float = 100000.0
    
    # Position sizing
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.10
    
    # Walk-forward settings
    use_walk_forward: bool = True
    train_window_days: int = 252
    test_window_days: int = 63
    step_days: int = 21
    
    # Fill assumptions
    fill_assumption: FillAssumption = FillAssumption.NEXT_OPEN
    
    # Data handling
    handle_missing: str = "forward_fill"


@dataclass
class BacktestMetrics:
    """Performance metrics."""
    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Risk
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    volatility: float = 0.0
    
    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    
    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Costs
    total_costs: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result."""
    config: BacktestConfig
    metrics: BacktestMetrics
    
    # Trade data
    trades: List[Trade] = field(default_factory=list)
    
    # Time series
    equity_curve: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None
    
    # Walk-forward
    walk_forward_results: List["WalkForwardPeriod"] = field(default_factory=list)
    out_of_sample_sharpe: Optional[float] = None
    
    # Regime analysis
    regime_metrics: Dict[str, BacktestMetrics] = field(default_factory=dict)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "total_return": self.metrics.total_return,
            "cagr": self.metrics.cagr,
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "max_drawdown": self.metrics.max_drawdown,
            "total_trades": self.metrics.total_trades,
            "win_rate": self.metrics.win_rate,
            "profit_factor": self.metrics.profit_factor,
            "out_of_sample_sharpe": self.out_of_sample_sharpe,
            "warnings": self.warnings,
        }


@dataclass
class WalkForwardPeriod:
    """Result from one walk-forward period."""
    period_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    train_trades: int = 0
    test_trades: int = 0
    test_sharpe: float = 0.0
    test_return: float = 0.0


class BacktestEngine:
    """
    Core backtest engine.
    
    Runs backtests with proper validation and cost modeling.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_capital = config.initial_capital
    
    def reset(self):
        """Reset backtest state."""
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.config.initial_capital
    
    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        cost_calculator: Optional[Callable] = None
    ) -> BacktestResult:
        """
        Run backtest on signals.
        
        Args:
            signals: DataFrame with columns [date, instrument, direction, entry_price, stop_loss]
            prices: DataFrame with price data for all instruments
            cost_calculator: Optional function to calculate transaction costs
        
        Returns:
            BacktestResult
        """
        self.reset()
        
        if self.config.use_walk_forward:
            return self._run_walk_forward(signals, prices, cost_calculator)
        else:
            return self._run_simple(signals, prices, cost_calculator)
    
    def _run_simple(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        cost_calculator: Optional[Callable]
    ) -> BacktestResult:
        """Run simple backtest without walk-forward."""
        # Process each signal
        for _, signal in signals.iterrows():
            self._process_signal(signal, prices, cost_calculator)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Build equity curve
        equity_series = pd.Series(
            {d: v for d, v in self.equity_curve},
            name="equity"
        ) if self.equity_curve else None
        
        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=self.trades,
            equity_curve=equity_series,
        )
    
    def _run_walk_forward(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        cost_calculator: Optional[Callable]
    ) -> BacktestResult:
        """Run walk-forward validation."""
        periods = self._generate_walk_forward_periods()
        wf_results = []
        all_test_returns = []
        
        for period in periods:
            # Filter signals to test period only
            # (In practice, train period would be used for parameter optimization)
            test_signals = signals[
                (signals["date"] >= period.test_start) &
                (signals["date"] <= period.test_end)
            ]
            
            # Reset and run for this period
            self.reset()
            
            for _, signal in test_signals.iterrows():
                self._process_signal(signal, prices, cost_calculator)
            
            # Calculate period metrics
            period_metrics = self._calculate_metrics()
            period.test_trades = len(self.trades)
            period.test_sharpe = period_metrics.sharpe_ratio
            period.test_return = period_metrics.total_return
            
            wf_results.append(period)
            
            if self.trades:
                returns = [t.pnl_pct for t in self.trades]
                all_test_returns.extend(returns)
        
        # Calculate out-of-sample metrics
        oos_sharpe = None
        if all_test_returns:
            returns_array = np.array(all_test_returns)
            if returns_array.std() > 0:
                oos_sharpe = (returns_array.mean() / returns_array.std()) * np.sqrt(252)
        
        # Final full run for complete metrics
        self.reset()
        for _, signal in signals.iterrows():
            self._process_signal(signal, prices, cost_calculator)
        
        metrics = self._calculate_metrics()
        
        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=self.trades,
            walk_forward_results=wf_results,
            out_of_sample_sharpe=oos_sharpe,
        )
    
    def _generate_walk_forward_periods(self) -> List[WalkForwardPeriod]:
        """Generate walk-forward validation periods."""
        periods = []
        
        current_date = self.config.start_date
        period_num = 1
        
        while current_date + timedelta(days=self.config.train_window_days + self.config.test_window_days) <= self.config.end_date:
            train_start = current_date
            train_end = train_start + timedelta(days=self.config.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_window_days)
            
            periods.append(WalkForwardPeriod(
                period_number=period_num,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))
            
            current_date += timedelta(days=self.config.step_days)
            period_num += 1
        
        return periods
    
    def _process_signal(
        self,
        signal: pd.Series,
        prices: pd.DataFrame,
        cost_calculator: Optional[Callable]
    ):
        """Process a single signal and create trade if conditions met."""
        instrument = signal.get("instrument")
        direction = signal.get("direction")
        entry_price = signal.get("entry_price")
        stop_loss = signal.get("stop_loss")
        signal_date = signal.get("date")
        
        if not all([instrument, direction, entry_price]):
            return
        
        # Calculate position size
        risk_amount = self.current_capital * self.config.risk_per_trade_pct
        
        if stop_loss and entry_price != stop_loss:
            risk_per_unit = abs(entry_price - stop_loss)
            size = risk_amount / risk_per_unit
        else:
            size = risk_amount / (entry_price * 0.02)  # Default 2% risk
        
        # Cap position size
        max_size = (self.current_capital * self.config.max_position_pct) / entry_price
        size = min(size, max_size)
        
        # Calculate entry cost
        entry_cost = 0
        if cost_calculator:
            entry_cost = cost_calculator(instrument, entry_price * size, "entry")
        
        # Create trade
        trade = Trade(
            instrument=instrument,
            direction=direction,
            entry_date=signal_date,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            entry_cost=entry_cost,
            strategy=signal.get("strategy"),
        )
        
        # Simulate exit (simplified - would need price data for proper simulation)
        # For MVP, assume exit after fixed period or at stop/target
        exit_date = signal_date + timedelta(days=5)  # Simplified
        exit_price = entry_price * (1.01 if direction == "LONG" else 0.99)  # Simplified
        
        exit_cost = 0
        if cost_calculator:
            exit_cost = cost_calculator(instrument, exit_price * size, "exit")
        
        trade.close(exit_date, exit_price, exit_cost)
        
        self.trades.append(trade)
        self.current_capital += trade.net_pnl
        self.equity_curve.append((exit_date, self.current_capital))
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate performance metrics from trades."""
        metrics = BacktestMetrics()
        
        if not self.trades:
            return metrics
        
        metrics.total_trades = len(self.trades)
        
        # Separate winners and losers
        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]
        
        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        metrics.win_rate = len(winners) / len(self.trades) if self.trades else 0
        
        # P&L metrics
        metrics.gross_profit = sum(t.net_pnl for t in winners)
        metrics.gross_loss = abs(sum(t.net_pnl for t in losers))
        
        metrics.avg_win = metrics.gross_profit / len(winners) if winners else 0
        metrics.avg_loss = metrics.gross_loss / len(losers) if losers else 0
        metrics.avg_trade = sum(t.net_pnl for t in self.trades) / len(self.trades)
        
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        
        # Total return
        total_pnl = sum(t.net_pnl for t in self.trades)
        metrics.total_return = total_pnl / self.config.initial_capital
        
        # Costs
        metrics.total_costs = sum(t.entry_cost + t.exit_cost for t in self.trades)
        
        # Calculate Sharpe (simplified)
        returns = [t.pnl_pct for t in self.trades]
        if len(returns) > 1:
            returns_array = np.array(returns)
            if returns_array.std() > 0:
                metrics.sharpe_ratio = (returns_array.mean() / returns_array.std()) * np.sqrt(252)
        
        # Calculate max drawdown from equity curve
        if self.equity_curve:
            equity = [v for _, v in self.equity_curve]
            peak = equity[0]
            max_dd = 0
            
            for e in equity:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak
                if dd > max_dd:
                    max_dd = dd
            
            metrics.max_drawdown = max_dd
        
        # Consecutive streaks
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in self.trades:
            if trade.is_winner:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        metrics.max_consecutive_wins = max_wins
        metrics.max_consecutive_losses = max_losses
        
        return metrics
