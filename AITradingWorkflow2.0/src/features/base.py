"""
Base Features Module

Common feature calculations used across all asset classes:
- Returns (various periods)
- Momentum scores
- Volatility measures
- Technical indicators
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


# =============================================================================
# Return Calculations
# =============================================================================

def compute_returns(
    prices: pd.Series,
    periods: List[int] = [1, 5, 20, 60, 120]
) -> pd.DataFrame:
    """
    Compute returns over multiple periods.
    
    Args:
        prices: Price series
        periods: List of lookback periods in days
    
    Returns:
        DataFrame with return columns for each period
    """
    returns = pd.DataFrame(index=prices.index)
    
    for period in periods:
        col_name = f"return_{period}d"
        returns[col_name] = prices.pct_change(period)
    
    return returns


def compute_log_returns(
    prices: pd.Series,
    periods: List[int] = [1, 5, 20, 60, 120]
) -> pd.DataFrame:
    """
    Compute log returns over multiple periods.
    
    Args:
        prices: Price series
        periods: List of lookback periods
    
    Returns:
        DataFrame with log return columns
    """
    returns = pd.DataFrame(index=prices.index)
    
    for period in periods:
        col_name = f"log_return_{period}d"
        returns[col_name] = np.log(prices / prices.shift(period))
    
    return returns


def compute_return_zscore(
    prices: pd.Series,
    return_period: int = 20,
    lookback: int = 252
) -> pd.Series:
    """
    Compute z-score of returns vs rolling mean/std.
    
    Args:
        prices: Price series
        return_period: Period for return calculation
        lookback: Lookback for mean/std calculation
    
    Returns:
        Z-score series
    """
    returns = prices.pct_change(return_period)
    rolling_mean = returns.rolling(lookback).mean()
    rolling_std = returns.rolling(lookback).std()
    
    zscore = (returns - rolling_mean) / rolling_std
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    
    return zscore


# =============================================================================
# Momentum Features
# =============================================================================

def compute_momentum_score(
    prices: pd.Series,
    periods: List[int] = [20, 60, 120],
    lookback: int = 252
) -> pd.Series:
    """
    Compute composite momentum score as average of z-scored returns.
    
    Args:
        prices: Price series
        periods: Return periods to use (e.g., 1mo, 3mo, 6mo)
        lookback: Lookback for z-score calculation
    
    Returns:
        Momentum score series
    """
    zscores = []
    
    for period in periods:
        zscore = compute_return_zscore(prices, period, lookback)
        zscores.append(zscore)
    
    # Average of z-scores
    momentum = pd.concat(zscores, axis=1).mean(axis=1)
    
    return momentum


def compute_trend_strength(
    prices: pd.Series,
    short_period: int = 20,
    long_period: int = 60
) -> pd.Series:
    """
    Compute trend strength based on price vs moving averages.
    
    Returns value between -1 and 1:
    - Positive: price above both MAs, bullish
    - Negative: price below both MAs, bearish
    - Near zero: mixed/choppy
    
    Args:
        prices: Price series
        short_period: Short MA period
        long_period: Long MA period
    
    Returns:
        Trend strength series
    """
    ma_short = prices.rolling(short_period).mean()
    ma_long = prices.rolling(long_period).mean()
    
    # Distance from MAs as % of price
    dist_short = (prices - ma_short) / prices
    dist_long = (prices - ma_long) / prices
    
    # Combine: both positive = strong bullish, both negative = strong bearish
    trend = (np.sign(dist_short) + np.sign(dist_long)) / 2
    
    # Add magnitude
    magnitude = (np.abs(dist_short) + np.abs(dist_long)) / 2
    trend_strength = trend * np.clip(magnitude * 10, 0, 1)  # Scale magnitude
    
    return trend_strength


def compute_price_vs_ma(
    prices: pd.Series,
    ma_periods: List[int] = [20, 50, 200]
) -> pd.DataFrame:
    """
    Compute price position relative to moving averages.
    
    Args:
        prices: Price series
        ma_periods: MA periods to compute
    
    Returns:
        DataFrame with columns for each MA comparison
    """
    result = pd.DataFrame(index=prices.index)
    
    for period in ma_periods:
        ma = prices.rolling(period).mean()
        result[f"price_vs_ma{period}"] = (prices - ma) / ma
        result[f"above_ma{period}"] = (prices > ma).astype(int)
    
    return result


# =============================================================================
# Volatility Features
# =============================================================================

def compute_realized_volatility(
    prices: pd.Series,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Compute realized volatility from price series.
    
    Args:
        prices: Price series
        window: Rolling window for volatility
        annualize: Whether to annualize the volatility
        trading_days: Trading days per year for annualization
    
    Returns:
        Volatility series
    """
    returns = prices.pct_change()
    vol = returns.rolling(window).std()
    
    if annualize:
        vol = vol * np.sqrt(trading_days)
    
    return vol


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Compute Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
    
    Returns:
        ATR series
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr


def compute_volatility_percentile(
    prices: pd.Series,
    vol_window: int = 20,
    percentile_window: int = 252
) -> pd.Series:
    """
    Compute current volatility percentile vs history.
    
    Args:
        prices: Price series
        vol_window: Window for volatility calculation
        percentile_window: Window for percentile calculation
    
    Returns:
        Percentile (0-100) series
    """
    vol = compute_realized_volatility(prices, vol_window, annualize=False)
    
    def rolling_percentile(x):
        return (x.iloc[-1] <= x).sum() / len(x) * 100
    
    percentile = vol.rolling(percentile_window).apply(rolling_percentile)
    
    return percentile


def compute_volatility_regime(
    prices: pd.Series,
    vol_window: int = 20,
    percentile_window: int = 252,
    low_threshold: float = 25,
    high_threshold: float = 75
) -> pd.Series:
    """
    Classify volatility regime.
    
    Args:
        prices: Price series
        vol_window: Volatility window
        percentile_window: Percentile window
        low_threshold: Below this = LOW_VOL
        high_threshold: Above this = HIGH_VOL
    
    Returns:
        Regime series ('LOW_VOL', 'NORMAL', 'HIGH_VOL')
    """
    percentile = compute_volatility_percentile(prices, vol_window, percentile_window)
    
    regime = pd.Series(index=prices.index, dtype=str)
    regime[percentile <= low_threshold] = "LOW_VOL"
    regime[(percentile > low_threshold) & (percentile < high_threshold)] = "NORMAL"
    regime[percentile >= high_threshold] = "HIGH_VOL"
    
    return regime


# =============================================================================
# Technical Indicators
# =============================================================================

def compute_rsi(
    prices: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        period: RSI period
    
    Returns:
        RSI series (0-100)
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    Compute MACD indicator.
    
    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
    
    Returns:
        DataFrame with MACD, signal, and histogram
    """
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    
    return pd.DataFrame({
        "macd": macd,
        "macd_signal": signal,
        "macd_histogram": histogram
    })


def compute_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Compute Bollinger Bands.
    
    Args:
        prices: Price series
        period: MA period
        num_std: Number of standard deviations
    
    Returns:
        DataFrame with upper, middle, lower bands and %B
    """
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    # %B: where price is within bands (0 = lower, 1 = upper)
    pct_b = (prices - lower) / (upper - lower)
    
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_pct_b": pct_b
    })


# =============================================================================
# Feature Aggregation
# =============================================================================

def compute_all_base_features(
    prices: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute all base features for a price series.
    
    Args:
        prices: Close/Last prices
        high: High prices (optional, for ATR)
        low: Low prices (optional, for ATR)
    
    Returns:
        DataFrame with all computed features
    """
    features = pd.DataFrame(index=prices.index)
    
    # Returns
    returns = compute_returns(prices, [1, 5, 20, 60, 120])
    features = pd.concat([features, returns], axis=1)
    
    # Log returns
    log_returns = compute_log_returns(prices, [1, 5, 20, 60])
    features = pd.concat([features, log_returns], axis=1)
    
    # Momentum
    features["momentum_score"] = compute_momentum_score(prices)
    features["trend_strength"] = compute_trend_strength(prices)
    
    # Price vs MAs
    ma_features = compute_price_vs_ma(prices)
    features = pd.concat([features, ma_features], axis=1)
    
    # Volatility
    features["volatility_20d"] = compute_realized_volatility(prices, 20)
    features["volatility_60d"] = compute_realized_volatility(prices, 60)
    features["vol_percentile"] = compute_volatility_percentile(prices)
    features["vol_regime"] = compute_volatility_regime(prices)
    
    # ATR (if high/low available)
    if high is not None and low is not None:
        features["atr_14"] = compute_atr(high, low, prices, 14)
        features["atr_20"] = compute_atr(high, low, prices, 20)
    
    # Technical indicators
    features["rsi_14"] = compute_rsi(prices, 14)
    
    macd = compute_macd(prices)
    features = pd.concat([features, macd], axis=1)
    
    bb = compute_bollinger_bands(prices)
    features = pd.concat([features, bb], axis=1)
    
    logger.info(f"Computed {len(features.columns)} base features")
    
    return features
