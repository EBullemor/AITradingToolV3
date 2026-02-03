"""
Strategy Pods Module

Contains all trading strategy implementations:
- FX Carry + Momentum (Pod 1)
- BTC Trend + Volatility (Pod 2) - coming soon
- Commodities Term Structure (Pod 3) - coming soon
- Cross-Asset Risk (Pod 4) - coming soon
- Mean Reversion (Pod 5) - coming soon

Each strategy generates Signal objects that are fed to the aggregator.

Usage:
    from src.strategies import FXCarryMomentumStrategy, Signal
    
    # Create strategy
    strategy = FXCarryMomentumStrategy()
    
    # Generate signals
    signals = strategy.generate_signals(features, macro_data)
    
    # Use signals
    for signal in signals:
        print(signal.format_for_display())
"""

from .base import (
    Signal,
    SignalDirection,
    SignalStrength,
    SignalStatus,
    BaseStrategy,
    load_strategy_config,
)

from .fx_carry_momentum import (
    FXCarryMomentumStrategy,
    create_fx_carry_momentum_strategy,
)


__all__ = [
    # Base classes
    "Signal",
    "SignalDirection",
    "SignalStrength",
    "SignalStatus",
    "BaseStrategy",
    "load_strategy_config",
    
    # Strategies
    "FXCarryMomentumStrategy",
    "create_fx_carry_momentum_strategy",
]
