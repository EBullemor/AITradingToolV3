"""
Crypto/BTC Features Module

Features specific to Bitcoin and cryptocurrency trading:
- On-chain metrics
- Exchange flows
- Valuation ratios (MVRV)
- Network activity
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .base import (
    compute_returns,
    compute_momentum_score,
    compute_realized_volatility,
    compute_all_base_features,
)


# =============================================================================
# On-Chain Features
# =============================================================================

def compute_exchange_flow_signal(
    net_flow: pd.Series,
    lookback: int = 30
) -> pd.DataFrame:
    """
    Compute exchange flow signals.
    
    Negative flow (outflow) = accumulation = bullish
    Positive flow (inflow) = distribution = bearish
    
    Args:
        net_flow: Net exchange flow (positive = inflow)
        lookback: Lookback for z-score
    
    Returns:
        DataFrame with flow metrics
    """
    result = pd.DataFrame(index=net_flow.index)
    
    # Raw flow
    result["exchange_net_flow"] = net_flow
    
    # Cumulative flow
    result["cumulative_flow_30d"] = net_flow.rolling(30).sum()
    
    # Z-score
    mean = net_flow.rolling(lookback * 3).mean()
    std = net_flow.rolling(lookback * 3).std()
    result["flow_zscore"] = (net_flow - mean) / std
    
    # Signal: strong outflow = bullish
    result["flow_signal"] = -result["flow_zscore"].clip(-2, 2) / 2
    
    return result


def compute_mvrv_signal(
    mvrv: pd.Series,
    oversold_threshold: float = 1.0,
    overbought_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Compute MVRV (Market Value to Realized Value) signals.
    
    MVRV < 1: Market value below cost basis (oversold)
    MVRV > 3: Market value well above cost basis (overbought)
    
    Args:
        mvrv: MVRV ratio series
        oversold_threshold: Below this = oversold
        overbought_threshold: Above this = overbought
    
    Returns:
        DataFrame with MVRV metrics
    """
    result = pd.DataFrame(index=mvrv.index)
    
    result["mvrv"] = mvrv
    
    # Z-score vs history
    mean = mvrv.rolling(365).mean()
    std = mvrv.rolling(365).std()
    result["mvrv_zscore"] = (mvrv - mean) / std
    
    # Percentile
    def rolling_percentile(x):
        return (x.iloc[-1] <= x).sum() / len(x) * 100
    result["mvrv_percentile"] = mvrv.rolling(365).apply(rolling_percentile)
    
    # Valuation regime
    regime = pd.Series(index=mvrv.index, dtype=str)
    regime[mvrv < oversold_threshold] = "UNDERVALUED"
    regime[(mvrv >= oversold_threshold) & (mvrv <= overbought_threshold)] = "FAIR"
    regime[mvrv > overbought_threshold] = "OVERVALUED"
    result["valuation_regime"] = regime
    
    # Signal: buy when undervalued, sell when overvalued
    signal = pd.Series(0.0, index=mvrv.index)
    signal[mvrv < oversold_threshold] = (oversold_threshold - mvrv[mvrv < oversold_threshold])
    signal[mvrv > overbought_threshold] = -(mvrv[mvrv > overbought_threshold] - overbought_threshold)
    result["mvrv_signal"] = signal.clip(-1, 1)
    
    return result


def compute_sopr_signal(
    sopr: pd.Series,
    ma_period: int = 7
) -> pd.DataFrame:
    """
    Compute SOPR (Spent Output Profit Ratio) signals.
    
    SOPR < 1: Holders selling at a loss (capitulation)
    SOPR > 1: Holders selling at a profit (distribution)
    
    Args:
        sopr: SOPR series
        ma_period: Moving average period for smoothing
    
    Returns:
        DataFrame with SOPR metrics
    """
    result = pd.DataFrame(index=sopr.index)
    
    result["sopr"] = sopr
    result["sopr_ma"] = sopr.rolling(ma_period).mean()
    
    # Distance from 1 (break-even)
    result["sopr_deviation"] = sopr - 1
    
    # Z-score
    mean = sopr.rolling(90).mean()
    std = sopr.rolling(90).std()
    result["sopr_zscore"] = (sopr - mean) / std
    
    # Signal: capitulation (SOPR < 1) can be buying opportunity
    # But needs to be combined with other factors
    result["sopr_signal"] = -(result["sopr_deviation"]).clip(-0.1, 0.1) * 10
    
    return result


def compute_network_activity_signal(
    active_addresses: pd.Series,
    lookback: int = 30
) -> pd.DataFrame:
    """
    Compute network activity signals.
    
    Growing active addresses = adoption/bullish
    Declining active addresses = declining interest
    
    Args:
        active_addresses: Daily active address count
        lookback: Lookback for momentum
    
    Returns:
        DataFrame with activity metrics
    """
    result = pd.DataFrame(index=active_addresses.index)
    
    result["active_addresses"] = active_addresses
    
    # Growth rate
    result["address_growth_30d"] = active_addresses.pct_change(30)
    result["address_growth_90d"] = active_addresses.pct_change(90)
    
    # Z-score of level
    mean = active_addresses.rolling(365).mean()
    std = active_addresses.rolling(365).std()
    result["address_zscore"] = (active_addresses - mean) / std
    
    # Momentum z-score
    growth_mean = result["address_growth_30d"].rolling(180).mean()
    growth_std = result["address_growth_30d"].rolling(180).std()
    result["growth_zscore"] = (result["address_growth_30d"] - growth_mean) / growth_std
    
    # Signal: strong growth = bullish
    result["activity_signal"] = result["growth_zscore"].clip(-2, 2) / 2
    
    return result


# =============================================================================
# BTC-Specific Technical Features
# =============================================================================

def compute_btc_halving_cycle(
    prices: pd.Series,
    last_halving: datetime = datetime(2024, 4, 20)  # Approximate
) -> pd.DataFrame:
    """
    Compute halving cycle position.
    
    BTC tends to follow 4-year cycles around halvings.
    
    Args:
        prices: BTC prices
        last_halving: Date of last halving
    
    Returns:
        DataFrame with cycle metrics
    """
    result = pd.DataFrame(index=prices.index)
    
    # Days since halving
    halving_date = pd.Timestamp(last_halving)
    result["days_since_halving"] = (prices.index - halving_date).days
    
    # Cycle position (0-1, where 1 = next halving expected)
    cycle_length = 4 * 365  # ~4 years
    result["cycle_position"] = (result["days_since_halving"] % cycle_length) / cycle_length
    
    # Historical pattern suggests:
    # 0-0.25: Post-halving accumulation (bullish)
    # 0.25-0.5: Bull run
    # 0.5-0.75: Distribution/top
    # 0.75-1.0: Bear market
    
    cycle_phase = pd.Series(index=prices.index, dtype=str)
    cycle_phase[result["cycle_position"] <= 0.25] = "ACCUMULATION"
    cycle_phase[(result["cycle_position"] > 0.25) & (result["cycle_position"] <= 0.5)] = "BULL_RUN"
    cycle_phase[(result["cycle_position"] > 0.5) & (result["cycle_position"] <= 0.75)] = "DISTRIBUTION"
    cycle_phase[result["cycle_position"] > 0.75] = "BEAR"
    result["cycle_phase"] = cycle_phase
    
    return result


def compute_btc_dominance_signal(
    btc_price: pd.Series,
    btc_dominance: pd.Series,
    lookback: int = 30
) -> pd.DataFrame:
    """
    Compute BTC dominance signals.
    
    Rising dominance + rising price = strong BTC
    Falling dominance + rising price = alt season
    
    Args:
        btc_price: BTC price
        btc_dominance: BTC market cap dominance (%)
        lookback: Lookback period
    
    Returns:
        DataFrame with dominance metrics
    """
    result = pd.DataFrame(index=btc_price.index)
    
    result["btc_dominance"] = btc_dominance
    result["dominance_change"] = btc_dominance.diff(lookback)
    result["price_change"] = btc_price.pct_change(lookback)
    
    # Regime
    regime = pd.Series(index=btc_price.index, dtype=str)
    
    rising_dom = result["dominance_change"] > 0
    rising_price = result["price_change"] > 0
    
    regime[rising_dom & rising_price] = "BTC_STRENGTH"
    regime[~rising_dom & rising_price] = "ALT_SEASON"
    regime[rising_dom & ~rising_price] = "FLIGHT_TO_SAFETY"
    regime[~rising_dom & ~rising_price] = "CRYPTO_WEAKNESS"
    
    result["dominance_regime"] = regime
    
    return result


# =============================================================================
# Volatility Breakout Features
# =============================================================================

def compute_vol_breakout_signal(
    prices: pd.Series,
    vol_window: int = 20,
    breakout_percentile: float = 90
) -> pd.DataFrame:
    """
    Detect volatility breakouts.
    
    High volatility after low volatility often signals new trend.
    
    Args:
        prices: BTC prices
        vol_window: Window for volatility calculation
        breakout_percentile: Percentile for breakout threshold
    
    Returns:
        DataFrame with breakout metrics
    """
    result = pd.DataFrame(index=prices.index)
    
    # Realized volatility
    returns = prices.pct_change()
    vol = returns.rolling(vol_window).std() * np.sqrt(365)
    result["realized_vol"] = vol
    
    # Vol percentile
    def rolling_percentile(x):
        return (x.iloc[-1] <= x).sum() / len(x) * 100
    result["vol_percentile"] = vol.rolling(252).apply(rolling_percentile)
    
    # Vol regime
    result["vol_low"] = result["vol_percentile"] < 25
    result["vol_high"] = result["vol_percentile"] > breakout_percentile
    
    # Breakout: transition from low to high vol
    result["vol_expanding"] = vol > vol.shift(5) * 1.5
    
    # Breakout signal
    result["breakout_signal"] = (
        result["vol_high"] & 
        result["vol_expanding"]
    ).astype(int)
    
    # Direction hint from price
    price_direction = np.sign(prices.pct_change(5))
    result["breakout_direction"] = result["breakout_signal"] * price_direction
    
    return result


# =============================================================================
# Combined BTC Features
# =============================================================================

def compute_btc_features(
    prices: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    exchange_flow: Optional[pd.Series] = None,
    mvrv: Optional[pd.Series] = None,
    sopr: Optional[pd.Series] = None,
    active_addresses: Optional[pd.Series] = None,
    btc_dominance: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute all BTC features.
    
    Args:
        prices: BTC prices
        high: High prices
        low: Low prices
        exchange_flow: Net exchange flow
        mvrv: MVRV ratio
        sopr: SOPR ratio
        active_addresses: Active address count
        btc_dominance: BTC market dominance (%)
    
    Returns:
        DataFrame with all BTC features
    """
    # Base features
    features = compute_all_base_features(prices, high, low)
    
    # On-chain features
    if exchange_flow is not None:
        flow_features = compute_exchange_flow_signal(exchange_flow)
        features = pd.concat([features, flow_features], axis=1)
    
    if mvrv is not None:
        mvrv_features = compute_mvrv_signal(mvrv)
        features = pd.concat([features, mvrv_features], axis=1)
    
    if sopr is not None:
        sopr_features = compute_sopr_signal(sopr)
        features = pd.concat([features, sopr_features], axis=1)
    
    if active_addresses is not None:
        activity_features = compute_network_activity_signal(active_addresses)
        features = pd.concat([features, activity_features], axis=1)
    
    # Halving cycle
    cycle_features = compute_btc_halving_cycle(prices)
    features = pd.concat([features, cycle_features], axis=1)
    
    # Vol breakout
    breakout_features = compute_vol_breakout_signal(prices)
    features = pd.concat([features, breakout_features], axis=1)
    
    # Dominance
    if btc_dominance is not None:
        dom_features = compute_btc_dominance_signal(prices, btc_dominance)
        features = pd.concat([features, dom_features], axis=1)
    
    features["ticker"] = "BTCUSD"
    
    logger.info(f"Computed {len(features.columns)} BTC features")
    
    return features


def compute_btc_composite_signal(
    features: pd.DataFrame,
    trend_weight: float = 0.3,
    onchain_weight: float = 0.3,
    valuation_weight: float = 0.2,
    breakout_weight: float = 0.2
) -> pd.Series:
    """
    Generate composite BTC signal.
    
    Args:
        features: BTC features DataFrame
        trend_weight: Weight for trend/momentum
        onchain_weight: Weight for on-chain metrics
        valuation_weight: Weight for valuation (MVRV)
        breakout_weight: Weight for vol breakout
    
    Returns:
        Composite signal (-1 to 1)
    """
    signal = pd.Series(0.0, index=features.index)
    
    # Trend component
    if "momentum_score" in features.columns:
        trend_signal = features["momentum_score"].clip(-2, 2) / 2
        signal += trend_signal * trend_weight
    
    # On-chain component
    onchain_signals = []
    if "flow_signal" in features.columns:
        onchain_signals.append(features["flow_signal"])
    if "activity_signal" in features.columns:
        onchain_signals.append(features["activity_signal"])
    
    if onchain_signals:
        onchain_avg = pd.concat(onchain_signals, axis=1).mean(axis=1)
        signal += onchain_avg * onchain_weight
    
    # Valuation component
    if "mvrv_signal" in features.columns:
        signal += features["mvrv_signal"] * valuation_weight
    
    # Breakout component
    if "breakout_direction" in features.columns:
        signal += features["breakout_direction"] * breakout_weight
    
    return signal.clip(-1, 1)
