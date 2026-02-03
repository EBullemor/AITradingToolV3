"""
Commodities Features Module

Features specific to commodity trading:
- Term structure (backwardation/contango)
- Roll yield
- Inventory analysis
- Seasonality
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
# Term Structure Features
# =============================================================================

def compute_roll_yield(
    front_price: pd.Series,
    back_price: pd.Series,
    days_to_roll: int = 30
) -> pd.Series:
    """
    Compute annualized roll yield from term structure.
    
    Positive roll yield = backwardation (bullish)
    Negative roll yield = contango (bearish for long)
    
    Args:
        front_price: Front month futures price
        back_price: Back month (e.g., 4th month) futures price
        days_to_roll: Days between contract months
    
    Returns:
        Annualized roll yield (%)
    """
    # Roll yield = (front - back) / back, annualized
    spread = front_price - back_price
    roll_yield = (spread / back_price) * (365 / days_to_roll) * 100
    
    return roll_yield


def compute_term_structure_slope(
    front_price: pd.Series,
    back_price: pd.Series
) -> pd.Series:
    """
    Compute term structure slope (front vs back).
    
    Args:
        front_price: Front month price
        back_price: Back month price
    
    Returns:
        Term structure slope (positive = backwardation)
    """
    slope = (front_price - back_price) / back_price
    return slope


def compute_term_structure_regime(
    front_price: pd.Series,
    back_price: pd.Series,
    threshold: float = 0.02
) -> pd.Series:
    """
    Classify term structure regime.
    
    Args:
        front_price: Front month price
        back_price: Back month price
        threshold: Threshold for significant backwardation/contango
    
    Returns:
        Regime series ('BACKWARDATION', 'FLAT', 'CONTANGO')
    """
    slope = compute_term_structure_slope(front_price, back_price)
    
    regime = pd.Series(index=front_price.index, dtype=str)
    regime[slope > threshold] = "BACKWARDATION"
    regime[(slope >= -threshold) & (slope <= threshold)] = "FLAT"
    regime[slope < -threshold] = "CONTANGO"
    
    return regime


def compute_roll_yield_percentile(
    roll_yield: pd.Series,
    lookback: int = 252
) -> pd.Series:
    """
    Compute current roll yield percentile vs history.
    
    Args:
        roll_yield: Roll yield series
        lookback: Lookback period
    
    Returns:
        Percentile (0-100)
    """
    def rolling_percentile(x):
        return (x.iloc[-1] <= x).sum() / len(x) * 100
    
    percentile = roll_yield.rolling(lookback).apply(rolling_percentile)
    return percentile


# =============================================================================
# Inventory Features
# =============================================================================

def compute_inventory_zscore(
    inventory: pd.Series,
    lookback: int = 260  # ~5 years of weekly data
) -> pd.Series:
    """
    Compute inventory z-score vs historical average.
    
    Low inventory = bullish for prices
    High inventory = bearish for prices
    
    Args:
        inventory: Inventory levels (e.g., EIA crude stocks)
        lookback: Lookback period
    
    Returns:
        Z-score series
    """
    mean = inventory.rolling(lookback).mean()
    std = inventory.rolling(lookback).std()
    zscore = (inventory - mean) / std
    
    return zscore


def compute_inventory_change(
    inventory: pd.Series,
    periods: List[int] = [1, 4, 13]  # 1 week, 1 month, 1 quarter
) -> pd.DataFrame:
    """
    Compute inventory changes over various periods.
    
    Args:
        inventory: Inventory levels
        periods: Periods for change calculation
    
    Returns:
        DataFrame with inventory change metrics
    """
    result = pd.DataFrame(index=inventory.index)
    
    for period in periods:
        result[f"inv_change_{period}w"] = inventory.diff(period)
        result[f"inv_change_pct_{period}w"] = inventory.pct_change(period) * 100
    
    return result


def compute_inventory_surprise(
    actual: pd.Series,
    expected: pd.Series
) -> pd.Series:
    """
    Compute inventory surprise (actual vs consensus).
    
    Args:
        actual: Actual inventory change
        expected: Expected/consensus inventory change
    
    Returns:
        Surprise (actual - expected)
    """
    surprise = actual - expected
    return surprise


# =============================================================================
# Seasonality Features
# =============================================================================

def compute_seasonal_zscore(
    prices: pd.Series,
    period: str = "month"
) -> pd.Series:
    """
    Compute z-score vs seasonal average.
    
    Args:
        prices: Price series
        period: 'month' or 'week'
    
    Returns:
        Seasonal z-score
    """
    if period == "month":
        grouper = prices.index.month
    elif period == "week":
        grouper = prices.index.isocalendar().week
    else:
        raise ValueError(f"Unknown period: {period}")
    
    # Compute seasonal averages
    seasonal_mean = prices.groupby(grouper).transform("mean")
    seasonal_std = prices.groupby(grouper).transform("std")
    
    zscore = (prices - seasonal_mean) / seasonal_std
    
    return zscore


def compute_seasonal_pattern(
    prices: pd.Series,
    years: int = 5
) -> pd.DataFrame:
    """
    Compute historical seasonal pattern.
    
    Args:
        prices: Price series
        years: Years of history to use
    
    Returns:
        DataFrame with seasonal returns by month
    """
    # Filter to last N years
    cutoff = prices.index.max() - pd.DateOffset(years=years)
    recent = prices[prices.index >= cutoff]
    
    # Monthly returns
    monthly_returns = recent.resample("M").last().pct_change()
    
    # Group by month
    seasonal = monthly_returns.groupby(monthly_returns.index.month).agg(["mean", "std", "count"])
    seasonal.index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    
    return seasonal


def is_seasonal_favorable(
    current_date: datetime,
    seasonal_pattern: pd.DataFrame,
    threshold: float = 0.01
) -> bool:
    """
    Check if current month is seasonally favorable.
    
    Args:
        current_date: Current date
        seasonal_pattern: Seasonal pattern DataFrame
        threshold: Minimum avg return for favorable
    
    Returns:
        True if seasonally favorable
    """
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    current_month = month_names[current_date.month - 1]
    
    if current_month in seasonal_pattern.index:
        avg_return = seasonal_pattern.loc[current_month, ("mean", "mean")]
        return avg_return > threshold
    
    return False


# =============================================================================
# Commodity-Specific Momentum
# =============================================================================

def compute_commodity_momentum(
    prices: pd.Series,
    inventory: Optional[pd.Series] = None,
    roll_yield: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute commodity-specific momentum indicators.
    
    Args:
        prices: Commodity prices
        inventory: Inventory levels (optional)
        roll_yield: Roll yield (optional)
    
    Returns:
        DataFrame with momentum metrics
    """
    result = pd.DataFrame(index=prices.index)
    
    # Price momentum
    result["price_momentum_1m"] = prices.pct_change(20)
    result["price_momentum_3m"] = prices.pct_change(60)
    
    # Momentum z-scores
    for col in ["price_momentum_1m", "price_momentum_3m"]:
        mean = result[col].rolling(252).mean()
        std = result[col].rolling(252).std()
        result[f"{col}_zscore"] = (result[col] - mean) / std
    
    # Combined momentum score
    result["momentum_score"] = (
        result["price_momentum_1m_zscore"] + 
        result["price_momentum_3m_zscore"]
    ) / 2
    
    # Inventory momentum (if available)
    if inventory is not None:
        inv_change = inventory.pct_change(4)  # 4-week change
        result["inventory_momentum"] = -inv_change  # Negative = bullish
    
    # Roll momentum (if available)
    if roll_yield is not None:
        result["roll_momentum"] = roll_yield.diff(20)
    
    return result


# =============================================================================
# Combined Commodity Features
# =============================================================================

def compute_commodity_features(
    prices: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    front_price: Optional[pd.Series] = None,
    back_price: Optional[pd.Series] = None,
    inventory: Optional[pd.Series] = None,
    ticker: str = "CL"
) -> pd.DataFrame:
    """
    Compute all commodity features.
    
    Args:
        prices: Commodity prices
        high: High prices
        low: Low prices
        front_price: Front month price (for term structure)
        back_price: Back month price
        inventory: Inventory levels
        ticker: Commodity ticker
    
    Returns:
        DataFrame with all features
    """
    # Base features
    features = compute_all_base_features(prices, high, low)
    
    # Term structure features
    if front_price is not None and back_price is not None:
        features["roll_yield"] = compute_roll_yield(front_price, back_price)
        features["term_structure_slope"] = compute_term_structure_slope(front_price, back_price)
        features["term_structure_regime"] = compute_term_structure_regime(front_price, back_price)
        features["roll_yield_percentile"] = compute_roll_yield_percentile(features["roll_yield"])
    
    # Inventory features
    if inventory is not None:
        features["inventory_zscore"] = compute_inventory_zscore(inventory)
        inv_changes = compute_inventory_change(inventory)
        features = pd.concat([features, inv_changes], axis=1)
    
    # Commodity momentum
    roll_yield = features.get("roll_yield")
    momentum = compute_commodity_momentum(prices, inventory, roll_yield)
    features = pd.concat([features, momentum], axis=1)
    
    # Seasonality
    features["seasonal_zscore"] = compute_seasonal_zscore(prices)
    
    # Ticker
    features["ticker"] = ticker
    
    logger.info(f"Computed {len(features.columns)} commodity features for {ticker}")
    
    return features


def compute_commodity_signal(
    features: pd.DataFrame,
    term_structure_weight: float = 0.3,
    momentum_weight: float = 0.3,
    inventory_weight: float = 0.2,
    seasonality_weight: float = 0.2
) -> pd.Series:
    """
    Generate commodity trading signal from features.
    
    Args:
        features: Commodity features DataFrame
        term_structure_weight: Weight for term structure
        momentum_weight: Weight for momentum
        inventory_weight: Weight for inventory
        seasonality_weight: Weight for seasonality
    
    Returns:
        Signal series (-1 to 1)
    """
    signal = pd.Series(0.0, index=features.index)
    
    # Term structure component
    if "roll_yield_percentile" in features.columns:
        ts_signal = (features["roll_yield_percentile"] - 50) / 50  # -1 to 1
        signal += ts_signal * term_structure_weight
    
    # Momentum component
    if "momentum_score" in features.columns:
        mom_signal = features["momentum_score"].clip(-2, 2) / 2  # Clip and normalize
        signal += mom_signal * momentum_weight
    
    # Inventory component (inverted - low inventory = bullish)
    if "inventory_zscore" in features.columns:
        inv_signal = -features["inventory_zscore"].clip(-2, 2) / 2
        signal += inv_signal * inventory_weight
    
    # Seasonality component
    if "seasonal_zscore" in features.columns:
        seas_signal = features["seasonal_zscore"].clip(-2, 2) / 2
        signal += seas_signal * seasonality_weight
    
    # Normalize to -1 to 1
    signal = signal.clip(-1, 1)
    
    return signal
