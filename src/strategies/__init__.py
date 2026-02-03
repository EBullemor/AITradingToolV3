"""
Strategy Pods Module

Contains all trading strategy implementations:
- FX Carry + Momentum (Pod 1)
- BTC Trend + Volatility (Pod 2)
- Commodities Term Structure (Pod 3)
- Cross-Asset Risk Sentiment (Pod 4)
- Mean Reversion (Pod 5)

Each strategy generates Signal objects that are fed to the aggregator.

Usage:
    from src.strategies import (
        FXCarryMomentumStrategy,
        BTCTrendVolStrategy,
        CommoditiesTSStrategy,
        CrossAssetRiskStrategy,
        MeanReversionStrategy,
        Signal,
    )
    
    # Create all strategies
    strategies = {
        "fx": FXCarryMomentumStrategy(),
        "btc": BTCTrendVolStrategy(),
        "commodities": CommoditiesTSStrategy(),
        "risk": CrossAssetRiskStrategy(),
        "mean_rev": MeanReversionStrategy(),
    }
    
    # Get portfolio-level risk allocation
    multiplier = strategies["risk"].get_allocation_multiplier(market_data)
    
    # Generate signals from all pods
    all_signals = []
    for name, strat in strategies.items():
        all_signals.extend(
            strat.generate_signals(features, macro_data)
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

from .mean_reversion import (
    MeanReversionStrategy,
    create_mean_reversion_strategy,
    SpikeDirection,
    CatalystVerdict,
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
    
    # Pod 5: Mean Reversion
    "MeanReversionStrategy",
    "create_mean_reversion_strategy",
    "SpikeDirection",
    "CatalystVerdict",
]
