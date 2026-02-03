"""
Feature Engineering Module

Provides feature calculations for all asset classes:
- Base features (returns, momentum, volatility, technicals)
- FX-specific features (carry, forward points)
- Commodity features (term structure, inventory)
- BTC/Crypto features (on-chain metrics, cycles)
- Regime detection (volatility, risk, trend)

Usage:
    from src.features import compute_fx_features, compute_market_regime
    
    # Compute FX features
    features = compute_fx_features(prices, pair="EURUSD")
    
    # Detect market regime
    regime = compute_market_regime(vix, prices)
"""

from .base import (
    # Returns
    compute_returns,
    compute_log_returns,
    compute_return_zscore,
    
    # Momentum
    compute_momentum_score,
    compute_trend_strength,
    compute_price_vs_ma,
    
    # Volatility
    compute_realized_volatility,
    compute_atr,
    compute_volatility_percentile,
    compute_volatility_regime,
    
    # Technicals
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    
    # Combined
    compute_all_base_features,
)

from .fx_features import (
    compute_carry_differential,
    compute_carry_score,
    compute_forward_implied_carry,
    compute_fx_momentum_score,
    compute_trend_following_signal,
    compute_dxy_beta,
    compute_fx_features,
    compute_fx_features_multi,
    generate_carry_signal,
)

from .commodity_features import (
    compute_roll_yield,
    compute_term_structure_slope,
    compute_term_structure_regime,
    compute_inventory_zscore,
    compute_seasonal_zscore,
    compute_commodity_features,
    compute_commodity_signal,
)

from .btc_features import (
    compute_exchange_flow_signal,
    compute_mvrv_signal,
    compute_sopr_signal,
    compute_network_activity_signal,
    compute_btc_halving_cycle,
    compute_vol_breakout_signal,
    compute_btc_features,
    compute_btc_composite_signal,
)

from .regime import (
    VolatilityRegime,
    RiskRegime,
    TrendRegime,
    compute_vix_regime,
    compute_vix_percentile_regime,
    compute_risk_regime,
    compute_trend_regime,
    compute_market_regime,
    compute_strategy_regime,
    detect_regime_changes,
)


__all__ = [
    # Base
    "compute_returns",
    "compute_log_returns",
    "compute_return_zscore",
    "compute_momentum_score",
    "compute_trend_strength",
    "compute_price_vs_ma",
    "compute_realized_volatility",
    "compute_atr",
    "compute_volatility_percentile",
    "compute_volatility_regime",
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_bands",
    "compute_all_base_features",
    
    # FX
    "compute_carry_differential",
    "compute_carry_score",
    "compute_forward_implied_carry",
    "compute_fx_momentum_score",
    "compute_trend_following_signal",
    "compute_dxy_beta",
    "compute_fx_features",
    "compute_fx_features_multi",
    "generate_carry_signal",
    
    # Commodities
    "compute_roll_yield",
    "compute_term_structure_slope",
    "compute_term_structure_regime",
    "compute_inventory_zscore",
    "compute_seasonal_zscore",
    "compute_commodity_features",
    "compute_commodity_signal",
    
    # BTC
    "compute_exchange_flow_signal",
    "compute_mvrv_signal",
    "compute_sopr_signal",
    "compute_network_activity_signal",
    "compute_btc_halving_cycle",
    "compute_vol_breakout_signal",
    "compute_btc_features",
    "compute_btc_composite_signal",
    
    # Regime
    "VolatilityRegime",
    "RiskRegime",
    "TrendRegime",
    "compute_vix_regime",
    "compute_vix_percentile_regime",
    "compute_risk_regime",
    "compute_trend_regime",
    "compute_market_regime",
    "compute_strategy_regime",
    "detect_regime_changes",
]
