"""
Strategy Pods Module

Contains all trading strategy implementations:
- FX Carry + Momentum (Pod 1)
- BTC Trend + Volatility (Pod 2)
- Commodities Term Structure (Pod 3)
- Cross-Asset Risk (Pod 4) - coming soon
- Mean Reversion (Pod 5) - coming soon

Each strategy generates Signal objects that are fed to the aggregator.

Usage:
    from src.strategies import (
        FXCarryMomentumStrategy,
        BTCTrendVolStrategy,
        CommoditiesTSStrategy,
        Signal,
    )
    
    # Create strategies
    fx_strategy = FXCarryMomentumStrategy()
    btc_strategy = BTCTrendVolStrategy()
    commod_strategy = CommoditiesTSStrategy()
    
    # Generate signals
    all_signals = (
        fx_strategy.generate_signals(features, macro_data)
        + btc_strategy.generate_signals(features, macro_data)
        + commod_strategy.generate_signals(features, macro_data)
    )
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

from .btc_trend_vol import (
    BTCTrendVolStrategy,
    create_btc_trend_vol_strategy,
)

from .commodities_ts import (
    CommoditiesTSStrategy,
    create_commodities_ts_strategy,
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
    "BTCTrendVolStrategy",
    "create_btc_trend_vol_strategy",
    "CommoditiesTSStrategy",
    "create_commodities_ts_strategy",
]
