"""
Strategy Pods Module

Contains all trading strategy implementations:
- FX Carry + Momentum (Pod 1) ✅
- BTC Trend + Volatility (Pod 2) - coming soon
- Commodities Term Structure (Pod 3) - coming soon
- Cross-Asset Risk (Pod 4) ✅
- Mean Reversion (Pod 5) - coming soon

Each strategy generates Signal objects that are fed to the aggregator.

Usage:
    from src.strategies import FXCarryMomentumStrategy, Signal
    from src.strategies import CrossAssetRiskOverlay

    # Create strategy
    strategy = FXCarryMomentumStrategy()
    overlay = CrossAssetRiskOverlay()

    # Check risk overlay before generating signals
    risk = overlay.compute_risk_score(vix=22.5, vix_history=vix_series)
    if risk.regime != "KILL":
        signals = strategy.generate_signals(features, macro_data)
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

from .cross_asset_risk import (
    CrossAssetRiskOverlay,
    RiskAssessment,
    CircuitBreakerResult,
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

    # Risk Overlay (Pod 4)
    "CrossAssetRiskOverlay",
    "RiskAssessment",
    "CircuitBreakerResult",
]
