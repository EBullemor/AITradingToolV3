"""
FX-Specific Features Module

Features specific to currency trading:
- Carry (interest rate differentials)
- Forward points analysis
- Cross-currency relationships
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
    compute_volatility_percentile,
    compute_all_base_features,
)


# =============================================================================
# Interest Rate / Carry Features
# =============================================================================

# Standard interest rate tickers (can be overridden)
RATE_TICKERS = {
    "USD": "US0003M",  # 3-month USD rate
    "EUR": "EUR003M",
    "JPY": "JPY003M",
    "GBP": "GBP003M",
    "AUD": "AUD003M",
    "CHF": "CHF003M",
    "CAD": "CAD003M",
    "NZD": "NZD003M",
}

# Currency pair to base/quote mapping
PAIR_CURRENCIES = {
    "EURUSD": ("EUR", "USD"),
    "USDJPY": ("USD", "JPY"),
    "GBPUSD": ("GBP", "USD"),
    "AUDUSD": ("AUD", "USD"),
    "USDCHF": ("USD", "CHF"),
    "USDCAD": ("USD", "CAD"),
    "NZDUSD": ("NZD", "USD"),
}


def compute_carry_differential(
    base_rate: pd.Series,
    quote_rate: pd.Series
) -> pd.Series:
    """
    Compute carry (interest rate differential).
    
    For a pair like EURUSD:
    - Positive carry = long position earns interest
    - Negative carry = long position pays interest
    
    Args:
        base_rate: Interest rate of base currency (e.g., EUR for EURUSD)
        quote_rate: Interest rate of quote currency (e.g., USD for EURUSD)
    
    Returns:
        Carry differential series (annualized %)
    """
    carry = base_rate - quote_rate
    return carry


def compute_carry_score(
    carry: pd.Series,
    volatility: pd.Series,
    min_vol: float = 0.01
) -> pd.Series:
    """
    Compute carry score (carry adjusted for volatility).
    
    Higher score = more attractive carry trade.
    
    Args:
        carry: Interest rate differential
        volatility: Realized volatility of the pair
        min_vol: Minimum volatility to prevent division issues
    
    Returns:
        Carry score series
    """
    vol_adj = volatility.clip(lower=min_vol)
    carry_score = carry / vol_adj
    
    return carry_score


def compute_forward_implied_carry(
    spot: pd.Series,
    forward_points: pd.Series,
    days_to_expiry: int = 30
) -> pd.Series:
    """
    Compute implied carry from forward points.
    
    Forward points reflect interest rate differential.
    
    Args:
        spot: Spot price
        forward_points: Forward points (in pips, divide by 10000 for majors)
        days_to_expiry: Days to forward expiry
    
    Returns:
        Annualized implied carry (%)
    """
    # Convert forward points to price terms
    # For most pairs, forward points are quoted in pips (0.0001)
    forward_price = spot + (forward_points / 10000)
    
    # Calculate implied carry (annualized)
    years = days_to_expiry / 365
    implied_carry = ((forward_price / spot) - 1) / years * 100
    
    return implied_carry


def compute_carry_momentum(
    carry: pd.Series,
    lookback: int = 60
) -> pd.Series:
    """
    Compute momentum in carry (is carry improving or deteriorating?).
    
    Args:
        carry: Carry series
        lookback: Lookback period
    
    Returns:
        Carry momentum (change in carry)
    """
    carry_change = carry - carry.shift(lookback)
    carry_zscore = (carry_change - carry_change.rolling(252).mean()) / carry_change.rolling(252).std()
    
    return carry_zscore


# =============================================================================
# FX-Specific Momentum Features
# =============================================================================

def compute_fx_momentum_score(
    prices: pd.Series,
    periods: List[int] = [20, 60, 120],
    weights: Optional[List[float]] = None
) -> pd.Series:
    """
    Compute FX-optimized momentum score.
    
    Uses weighted average of return z-scores with decay for older periods.
    
    Args:
        prices: FX spot prices
        periods: Return periods (typically 1mo, 3mo, 6mo)
        weights: Weights for each period (default: equal)
    
    Returns:
        Momentum score series
    """
    if weights is None:
        weights = [1.0] * len(periods)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    zscores = []
    for period in periods:
        returns = prices.pct_change(period)
        mean = returns.rolling(252).mean()
        std = returns.rolling(252).std()
        zscore = (returns - mean) / std
        zscores.append(zscore)
    
    # Weighted average
    momentum = sum(w * z for w, z in zip(weights, zscores))
    
    return momentum


def compute_trend_following_signal(
    prices: pd.Series,
    fast_ma: int = 20,
    slow_ma: int = 50,
    very_slow_ma: int = 200
) -> pd.DataFrame:
    """
    Compute trend-following signals for FX.
    
    Args:
        prices: FX spot prices
        fast_ma: Fast MA period
        slow_ma: Slow MA period
        very_slow_ma: Very slow MA period (200-day)
    
    Returns:
        DataFrame with trend signals
    """
    ma_fast = prices.rolling(fast_ma).mean()
    ma_slow = prices.rolling(slow_ma).mean()
    ma_very_slow = prices.rolling(very_slow_ma).mean()
    
    signals = pd.DataFrame(index=prices.index)
    
    # Binary signals
    signals["above_fast_ma"] = (prices > ma_fast).astype(int)
    signals["above_slow_ma"] = (prices > ma_slow).astype(int)
    signals["above_200ma"] = (prices > ma_very_slow).astype(int)
    signals["fast_above_slow"] = (ma_fast > ma_slow).astype(int)
    
    # Composite trend score (-1 to +1)
    signals["trend_score"] = (
        signals["above_fast_ma"] + 
        signals["above_slow_ma"] + 
        signals["above_200ma"] +
        signals["fast_above_slow"]
    ) / 4 * 2 - 1  # Scale to -1 to +1
    
    return signals


# =============================================================================
# Cross-Currency Features
# =============================================================================

def compute_dxy_beta(
    pair_prices: pd.Series,
    dxy_prices: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Compute rolling beta to DXY (USD strength).
    
    Args:
        pair_prices: FX pair prices
        dxy_prices: DXY index prices
        window: Rolling window for beta calculation
    
    Returns:
        Beta series
    """
    pair_returns = pair_prices.pct_change()
    dxy_returns = dxy_prices.pct_change()
    
    # Rolling covariance and variance
    covariance = pair_returns.rolling(window).cov(dxy_returns)
    variance = dxy_returns.rolling(window).var()
    
    beta = covariance / variance
    
    return beta


def compute_relative_strength(
    pair1_prices: pd.Series,
    pair2_prices: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Compute relative strength between two currency pairs.
    
    Useful for pairs trading or cross-currency analysis.
    
    Args:
        pair1_prices: First pair prices
        pair2_prices: Second pair prices
        period: Lookback period
    
    Returns:
        Relative strength (ratio momentum)
    """
    ratio = pair1_prices / pair2_prices
    rs = ratio.pct_change(period)
    
    return rs


# =============================================================================
# Combined FX Features
# =============================================================================

def compute_fx_features(
    prices: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    pair: str = "EURUSD",
    carry: Optional[pd.Series] = None,
    forward_points: Optional[pd.Series] = None,
    dxy: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute all FX features for a currency pair.
    
    Args:
        prices: Spot prices
        high: High prices (optional)
        low: Low prices (optional)
        pair: Currency pair name
        carry: Pre-computed carry differential (optional)
        forward_points: Forward points for implied carry (optional)
        dxy: DXY prices for beta calculation (optional)
    
    Returns:
        DataFrame with all FX features
    """
    # Start with base features
    features = compute_all_base_features(prices, high, low)
    
    # FX-specific momentum
    features["fx_momentum"] = compute_fx_momentum_score(prices)
    
    # Trend signals
    trend_signals = compute_trend_following_signal(prices)
    features = pd.concat([features, trend_signals], axis=1)
    
    # Carry features (if available)
    if carry is not None:
        features["carry"] = carry
        vol = compute_realized_volatility(prices, 20)
        features["carry_score"] = compute_carry_score(carry, vol)
        features["carry_momentum"] = compute_carry_momentum(carry)
    
    # Forward-implied carry (if available)
    if forward_points is not None:
        features["implied_carry_1m"] = compute_forward_implied_carry(prices, forward_points, 30)
    
    # DXY beta (if available)
    if dxy is not None:
        features["dxy_beta"] = compute_dxy_beta(prices, dxy)
    
    # Add pair identifier
    features["pair"] = pair
    
    logger.info(f"Computed {len(features.columns)} FX features for {pair}")
    
    return features


def compute_fx_features_multi(
    data: Dict[str, pd.DataFrame],
    pairs: List[str] = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"],
    dxy_data: Optional[pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compute FX features for multiple pairs.
    
    Args:
        data: Dictionary mapping pair to DataFrame with price columns
        pairs: List of pairs to process
        dxy_data: DXY data for cross-pair features
    
    Returns:
        Dictionary mapping pair to features DataFrame
    """
    all_features = {}
    
    dxy_prices = None
    if dxy_data is not None and "PX_LAST" in dxy_data.columns:
        dxy_prices = dxy_data["PX_LAST"]
    
    for pair in pairs:
        if pair not in data:
            logger.warning(f"No data for {pair}, skipping")
            continue
        
        df = data[pair]
        
        prices = df["PX_LAST"] if "PX_LAST" in df.columns else df.iloc[:, 0]
        high = df.get("PX_HIGH")
        low = df.get("PX_LOW")
        carry = df.get("carry")
        fwd_pts = df.get("FWD_POINTS_1M")
        
        features = compute_fx_features(
            prices=prices,
            high=high,
            low=low,
            pair=pair,
            carry=carry,
            forward_points=fwd_pts,
            dxy=dxy_prices
        )
        
        all_features[pair] = features
    
    return all_features


# =============================================================================
# Signal Generation Helpers
# =============================================================================

def generate_carry_signal(
    carry_score: pd.Series,
    momentum_score: pd.Series,
    vol_regime: pd.Series,
    carry_threshold: float = 0.5,
    momentum_threshold: float = 0.0
) -> pd.Series:
    """
    Generate carry trade signal (simplified version).
    
    Logic:
    - Long if carry_score > threshold AND momentum > 0 AND vol_regime != HIGH_VOL
    - Short if carry_score < -threshold AND momentum < 0 AND vol_regime != HIGH_VOL
    - Flat otherwise
    
    Args:
        carry_score: Volatility-adjusted carry
        momentum_score: Momentum indicator
        vol_regime: Volatility regime
        carry_threshold: Minimum carry score for signal
        momentum_threshold: Minimum momentum for signal
    
    Returns:
        Signal series (-1, 0, 1)
    """
    signal = pd.Series(0, index=carry_score.index)
    
    # Long conditions
    long_cond = (
        (carry_score > carry_threshold) &
        (momentum_score > momentum_threshold) &
        (vol_regime != "HIGH_VOL")
    )
    
    # Short conditions
    short_cond = (
        (carry_score < -carry_threshold) &
        (momentum_score < -momentum_threshold) &
        (vol_regime != "HIGH_VOL")
    )
    
    signal[long_cond] = 1
    signal[short_cond] = -1
    
    return signal
