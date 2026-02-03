"""
Strategy Pods Module

Contains all trading strategy implementations:
- FX Carry + Momentum (Pod 1)
- BTC Trend + Volatility (Pod 2)
- Commodities Term Structure (Pod 3)
- Cross-Asset Risk Sentiment (Pod 4)
- Mean Reversion (Pod 5) - coming soon

Each strategy generates Signal objects that are fed to the aggregator.

Usage:
    from src.strategies import (
        FXCarryMomentumStrategy,
        BTCTrendVolStrategy,
        CommoditiesTSStrategy,
        CrossAssetRiskStrategy,
        Signal,
    )
    
    # Create strategies
    fx_strategy = FXCarryMomentumStrategy()
    btc_strategy = BTCTrendVolStrategy()
    commod_strategy = CommoditiesTSStrategy()
    risk_strategy = CrossAssetRiskStrategy()
    
    # Get portfolio-level risk allocation
    multiplier = risk_strategy.get_allocation_multiplier(market_data)
    
    # Generate signals
    all_signals = (
        fx_strategy.generate_signals(features, macro_data)
        + btc_strategy.generate_signals(features, macro_data)
        + commod_strategy.generate_signals(features, macro_data)
        + risk_strategy.generate_signals(features, macro_data)
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

from .cross_asset_risk import (
    CrossAssetRiskStrategy,
    create_cross_asset_risk_strategy,
    RiskRegime,
    ALLOCATION_MULTIPLIERS,
    STRATEGY_BIAS,
)


__all__ = [
    # Base classes
    "Signal",
    "SignalDirection",
    "SignalStrength",
    "SignalStatus",
    "BaseStrategy",
    "load_strategy_config",
    
    # Pod 1: FX Carry + Momentum
    "FXCarryMomentumStrategy",
    "create_fx_carry_momentum_strategy",
    
    # Pod 2: BTC Trend + Volatility
    "BTCTrendVolStrategy",
    "create_btc_trend_vol_strategy",
    
    # Pod 3: Commodities Term Structure
    "CommoditiesTSStrategy",
    "create_commodities_ts_strategy",
    
    # Pod 4: Cross-Asset Risk Sentiment
    "CrossAssetRiskStrategy",
    "create_cross_asset_risk_strategy",
    "RiskRegime",
    "ALLOCATION_MULTIPLIERS",
    "STRATEGY_BIAS",
]
